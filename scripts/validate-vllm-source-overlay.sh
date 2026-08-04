#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 <image-directory>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$1" = /* ]]; then
  image_dir="$(realpath "$1")"
else
  image_dir="$(realpath "$repo_root/$1")"
fi

case "$image_dir" in
  "$repo_root"/images/*) ;;
  *)
    echo "Image directory must be below images/: $image_dir" >&2
    exit 1
    ;;
esac

dependency_lock="$image_dir/dependency.lock.json"
source_dir="$image_dir/source-overlay"
source_lock="$image_dir/source-overlay.sha256"
runtime_lock="$image_dir/runtime-overlay.sha256"

for required_path in "$dependency_lock" "$source_dir" "$source_lock" "$runtime_lock"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing source-build input: $required_path" >&2
    exit 1
  fi
done

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT

(
  cd "$source_dir"
  sha256sum --check --quiet "$source_lock"
  sha256sum --check --quiet "$runtime_lock"
  find . -type f -print | sort > "$temporary_dir/actual-source-files"
)

awk '{print $2}' "$source_lock" | sort > "$temporary_dir/locked-source-files"
if ! diff -u "$temporary_dir/locked-source-files" "$temporary_dir/actual-source-files"; then
  echo "source-overlay.sha256 must cover every source overlay file exactly once." >&2
  exit 1
fi

source_files="$(wc -l < "$source_lock" | tr -d ' ')"
runtime_files="$(wc -l < "$runtime_lock" | tr -d ' ')"
if [[ "$source_files" != "$(jq -r '.source_build.source_overlay.files' "$dependency_lock")" ]]; then
  echo "Source overlay file count does not match dependency.lock.json." >&2
  exit 1
fi
if [[ "$runtime_files" != "$(jq -r '.source_build.runtime_overlay.files' "$dependency_lock")" ]]; then
  echo "Runtime overlay file count does not match dependency.lock.json." >&2
  exit 1
fi

check_locked_digest() {
  local relative_path="$1"
  local expected_digest="$2"
  local actual_digest

  actual_digest="$(sha256sum "$image_dir/$relative_path" | awk '{print $1}')"
  if [[ "$actual_digest" != "$expected_digest" ]]; then
    echo "Digest mismatch for $relative_path: expected $expected_digest, got $actual_digest" >&2
    exit 1
  fi
}

check_locked_digest \
  "$(jq -r '.source_build.cumulative_patch.path' "$dependency_lock")" \
  "$(jq -r '.source_build.cumulative_patch.sha256' "$dependency_lock")"
check_locked_digest \
  "$(jq -r '.source_build.source_overlay.lock' "$dependency_lock")" \
  "$(jq -r '.source_build.source_overlay.lock_sha256' "$dependency_lock")"
check_locked_digest \
  "$(jq -r '.source_build.runtime_overlay.lock' "$dependency_lock")" \
  "$(jq -r '.source_build.runtime_overlay.lock_sha256' "$dependency_lock")"
check_locked_digest \
  "$(jq -r '.provenance.knownfix_manifest.path' "$dependency_lock")" \
  "$(jq -r '.provenance.knownfix_manifest.sha256' "$dependency_lock")"
check_locked_digest \
  "$(jq -r '.provenance.runtime_manifest.path' "$dependency_lock")" \
  "$(jq -r '.provenance.runtime_manifest.sha256' "$dependency_lock")"

echo "Validated vLLM source overlay: ${image_dir#"$repo_root"/}"
