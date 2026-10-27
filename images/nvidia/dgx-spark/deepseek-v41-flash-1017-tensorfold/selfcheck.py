#!/usr/bin/env python3
"""Self-check for the DS41F 1.2.5 / K2a TensorFold image (Petabridge).

Run at build time by the Dockerfile (--full) and available in the image as
/opt/ds41f/selfcheck.py. Python standard library only. It checks what can be
checked without a GPU: the dlopen()ed libraries by soname, the engine binaries
against their build receipt, the recipe file the launcher compares, and the
RoPE tables. --runtime additionally loads libcuda.so.1 (injected by the NVIDIA
container runtime; it does not exist at build time).
"""
import argparse
import ctypes
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/ds41f-tensorfold")
RECIPE = Path("/opt/ds41f/recipe")

# The recipe file at upstream tag v1.2.5 (b4e476d6). The launcher compares the
# sha256 of this exact file with BUILD.json's recipe_manifest_sha256.
K2A_JSON_SHA256 = "600c83c88d72f22506facc7cc1883f8d9e4439268b7e1d198efdbcb4536de9bc"
SOURCE_ARCHIVE_SHA256 = "8d4b2f808827fd247048082d122b8fab31b75e798b5ed86fd370103ef178469a"
# Petabridge's recipe k2a-pb1 (pb1/make_recipe.py): k2a plus the pb1/ overlays. BUILD.json names it.
PB1_JSON_SHA256 = "0d0c908bdcb9a1b55fb77b249be3e45190dd39c45179f2519e5c564d2405ac97"
PB1_SOURCE_TREE_SHA256 = "91dc76c8a6061558392e26f1fbc22e80a9330c343cf31ea285455fe51a0328be"

NCCL_VERSION_CODE = 23007  # 2.30.7

# tensorfold/rope-hashes.json["10000000"] at the same commit.
ROPE_ROWS = 10002432
ROPE_BYTES = ROPE_ROWS * 32 * 4
ROPE_SHA256 = {
    "rope-plain-cos.f32": "e03f5e3a1f51888ed7c45671f9e10fc6c12091b2076f11299a0ba8b4acc58b00",
    "rope-plain-sin.f32": "ee17e9620292b35a0cd4afb0500ee41c32fcf121ef6d9139ab0111caf5920fe5",
    "rope-compressed-cos.f32": "232935aeb2d027d53101a404f6cea701864a33e1193338d4bc892ddde1fdb668",
    "rope-compressed-sin.f32": "a3f7f728812908d3d73982aa01c417bdfa732d95d819ab38e475f6309b051e37",
}

failures = []


def check(ok, label):
    print(("ok    " if ok else "FAIL  ") + label)
    if not ok:
        failures.append(label)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(soname):
    try:
        return ctypes.CDLL(soname)
    except OSError as error:
        check(False, f"dlopen {soname}: {error}")
        return None


def check_libraries(runtime):
    for soname in ("libcublas.so.13", "libcublasLt.so.13", "libibverbs.so.1"):
        check(load(soname) is not None, f"dlopen {soname}")
    nccl = load("libnccl.so.2")
    if nccl is not None:
        version = ctypes.c_int(0)
        result = nccl.ncclGetVersion(ctypes.byref(version))
        check(result == 0 and version.value == NCCL_VERSION_CODE,
              f"ncclGetVersion == {NCCL_VERSION_CODE} (got {version.value}, result {result})")
    providers = glob.glob("/usr/lib/*/libibverbs/libmlx5-rdmav*.so")
    check(bool(providers), "rdma-core mlx5 provider present")
    if runtime:
        check(load("libcuda.so.1") is not None, "dlopen libcuda.so.1 (host driver)")


def check_engine():
    binaries = ROOT / "bin-1.2.5"
    receipt = json.loads((binaries / "BUILD.json").read_bytes())
    check(receipt.get("schema") == "ds41f.native-build.v1", "BUILD.json schema")
    check(receipt.get("recipe") == "k2a-pb1" and receipt.get("version") == "1.2.5", "BUILD.json recipe k2a-pb1, version 1.2.5")
    check(receipt.get("adaptive_learning") is True, "BUILD.json adaptive_learning")
    check(receipt.get("recipe_manifest_sha256") == PB1_JSON_SHA256, "BUILD.json recipe_manifest_sha256")
    check(receipt.get("source_tree_sha256") == PB1_SOURCE_TREE_SHA256, "BUILD.json source_tree_sha256")
    check(receipt.get("source_archive_sha256") == SOURCE_ARCHIVE_SHA256, "BUILD.json source_archive_sha256")
    check(receipt.get("model_requests") == 0, "BUILD.json model_requests == 0")
    check(set(receipt.get("binaries", {})) == {"tensorfold-native", "tf-dsv41-lanes"}, "BUILD.json binaries")
    for name, record in receipt.get("binaries", {}).items():
        path = binaries / name
        check(path.is_file() and sha256(path) == record["sha256"], f"{name} sha256 equals BUILD.json")
        ldd = subprocess.run(["ldd", str(path)], capture_output=True, text=True)
        check("not found" not in ldd.stdout, f"{name} resolves its shared libraries (ldd)")
    check(sha256(RECIPE / "tensorfold/variants/recipes/k2a.json") == K2A_JSON_SHA256, "recipes/k2a.json sha256")
    check(sha256(RECIPE / "tensorfold/variants/recipes/k2a-pb1.json") == PB1_JSON_SHA256, "recipes/k2a-pb1.json sha256")


def check_kit(full):
    kit = ROOT / "kit"
    for relative in ("aot/aot.json", "engram-token-map.json", "rope.json",
                     "vision/torch_fmha_sm120.cubin", "vision/gate_bias_vl.safetensors",
                     "models/TF351/config.json"):
        check((kit / relative).is_file(), f"kit/{relative} present")
    rope = json.loads((kit / "rope.json").read_bytes())
    check(rope.get("rows") == ROPE_ROWS and rope.get("half") == 32, f"kit/rope.json rows == {ROPE_ROWS} (10M pool)")
    for name, expected in ROPE_SHA256.items():
        path = kit / name
        check(path.is_file() and path.stat().st_size == ROPE_BYTES, f"{name} size {ROPE_BYTES}")
        check(rope.get(name) == expected, f"kit/rope.json records {name}")
        if full and path.is_file():
            check(sha256(path) == expected, f"{name} sha256")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also hash the 5.12 GB of RoPE tables")
    parser.add_argument("--runtime", action="store_true", help="also load libcuda.so.1 (needs the NVIDIA runtime)")
    args = parser.parse_args()
    check_libraries(args.runtime)
    check_engine()
    check_kit(args.full)
    if failures:
        print(f"{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
