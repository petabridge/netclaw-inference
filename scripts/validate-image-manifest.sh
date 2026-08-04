#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 <path-to-image.json>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
manifest_input="$1"

if [[ "$manifest_input" = /* ]]; then
  manifest_path="$manifest_input"
else
  manifest_path="$repo_root/$manifest_input"
fi

if [[ ! -f "$manifest_path" ]]; then
  echo "Image manifest does not exist: $manifest_input" >&2
  exit 1
fi

manifest_path="$(realpath "$manifest_path")"
case "$manifest_path" in
  "$repo_root"/images/*/image.json) ;;
  *)
    echo "Image manifests must be named image.json below images/: $manifest_path" >&2
    exit 1
    ;;
esac

jq -e '
  type == "object" and
  ((keys - [
    "$schema", "schema_version", "id", "description", "vendor",
    "hardware_family", "platform", "registry_repository", "build_enabled",
    "disabled_reason", "context", "dockerfile", "build_args"
  ]) | length == 0) and
  (.schema_version == 1) and
  (.id | type == "string" and test("^[a-z0-9]+(?:-[a-z0-9]+)*$")) and
  (.description | type == "string" and length > 0) and
  (.vendor == "nvidia" or .vendor == "amd") and
  (.hardware_family | type == "string" and test("^[a-z0-9]+(?:-[a-z0-9]+)*$")) and
  (.platform == "linux/arm64" or .platform == "linux/amd64") and
  (.registry_repository | type == "string" and test("^petabridge/[a-z0-9]+(?:[._-][a-z0-9]+)*$")) and
  (.build_enabled | type == "boolean") and
  ((.build_args // {}) | type == "object") and
  ([((.build_args // {}) | to_entries[]) |
    (.key | test("^[A-Z][A-Z0-9_]*$")) and
    (.value | type == "string" and test("^[A-Za-z0-9._:/+@=-]+$"))
  ] | all) and
  if .build_enabled then
    (.context | type == "string" and length > 0) and
    (.dockerfile | type == "string" and length > 0)
  else
    (.disabled_reason | type == "string" and length > 0)
  end
' "$manifest_path" >/dev/null || {
  echo "Invalid image manifest structure: $manifest_path" >&2
  exit 1
}

id="$(jq -r '.id' "$manifest_path")"
vendor="$(jq -r '.vendor' "$manifest_path")"
hardware_family="$(jq -r '.hardware_family' "$manifest_path")"
platform="$(jq -r '.platform' "$manifest_path")"
registry_repository="$(jq -r '.registry_repository' "$manifest_path")"
build_enabled="$(jq -r '.build_enabled' "$manifest_path")"

if [[ "$registry_repository" != "petabridge/$id" ]]; then
  echo "registry_repository must end with the manifest id: $id" >&2
  exit 1
fi

case "$vendor/$hardware_family/$platform" in
  nvidia/dgx-spark/linux/arm64)
    expected_prefix="$repo_root/images/nvidia/dgx-spark/"
    ;;
  amd/rocm/linux/amd64)
    expected_prefix="$repo_root/images/amd/rocm/"
    ;;
  *)
    echo "Unsupported vendor/hardware/platform combination: $vendor/$hardware_family/$platform" >&2
    exit 1
    ;;
esac

case "$manifest_path" in
  "$expected_prefix"*/image.json) ;;
  *)
    echo "Manifest path does not match its vendor and hardware family: $manifest_path" >&2
    exit 1
    ;;
esac

if [[ "$build_enabled" != "true" ]]; then
  echo "Validated disabled image manifest: ${manifest_path#"$repo_root"/}"
  exit 0
fi

validate_relative_path() {
  local value="$1"
  local label="$2"
  local component

  if [[ "$value" = /* || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "$label must be a safe repository-relative path: $value" >&2
    return 1
  fi

  IFS='/' read -r -a components <<< "$value"
  for component in "${components[@]}"; do
    if [[ "$component" == ".." || -z "$component" ]]; then
      echo "$label contains an unsafe path component: $value" >&2
      return 1
    fi
  done
}

context="$(jq -r '.context' "$manifest_path")"
dockerfile="$(jq -r '.dockerfile' "$manifest_path")"
validate_relative_path "$context" "context"
validate_relative_path "$dockerfile" "dockerfile"

context_path="$(realpath "$repo_root/$context")"
dockerfile_path="$(realpath "$repo_root/$dockerfile")"

if [[ ! -d "$context_path" ]]; then
  echo "Build context does not exist: $context" >&2
  exit 1
fi
if [[ ! -f "$dockerfile_path" ]]; then
  echo "Dockerfile does not exist: $dockerfile" >&2
  exit 1
fi
case "$dockerfile_path" in
  "$context_path"/*|"$context_path") ;;
  *)
    echo "Dockerfile must be inside its build context: $dockerfile" >&2
    exit 1
    ;;
esac

declare -A stage_aliases=()
while IFS= read -r from_line; do
  read -r -a tokens <<< "$from_line"
  image_index=1
  if [[ "${tokens[1]:-}" == --platform=* ]]; then
    image_index=2
  fi
  base_image="${tokens[$image_index]:-}"

  if [[ -z "$base_image" ]]; then
    echo "Malformed FROM instruction in $dockerfile: $from_line" >&2
    exit 1
  fi

  if [[ "$base_image" != "scratch" && -z "${stage_aliases[$base_image]:-}" &&
        ! "$base_image" =~ @sha256:[a-f0-9]{64}$ ]]; then
    echo "External base images must be pinned by sha256 digest: $from_line" >&2
    exit 1
  fi

  alias_index=$((image_index + 1))
  if [[ "${tokens[$alias_index]:-}" =~ ^([Aa][Ss])$ && -n "${tokens[$((alias_index + 1))]:-}" ]]; then
    stage_aliases["${tokens[$((alias_index + 1))]}"]=1
  fi
done < <(awk 'toupper($1) == "FROM" { print }' "$dockerfile_path")

if [[ ${#stage_aliases[@]} -eq 0 ]] && ! awk 'toupper($1) == "FROM" { found=1 } END { exit !found }' "$dockerfile_path"; then
  echo "Dockerfile has no FROM instruction: $dockerfile" >&2
  exit 1
fi

echo "Validated enabled image manifest: ${manifest_path#"$repo_root"/}"
