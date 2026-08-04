#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 <path-to-image.json> <candidate-tag>" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
manifest="$1"
candidate_tag="$2"
registry="${REGISTRY:-ghcr.io}"
metadata_file="${BUILD_METADATA_FILE:-$repo_root/build-metadata.json}"

if [[ ! "$candidate_tag" =~ ^sha-[a-f0-9]{12}$ ]]; then
  echo "Candidate tags must have the form sha-<12 lowercase hex characters>." >&2
  exit 1
fi
if [[ ! "$registry" =~ ^[a-zA-Z0-9.-]+(:[0-9]+)?$ ]]; then
  echo "REGISTRY must be a hostname with an optional port, without a URL scheme." >&2
  exit 1
fi

"$repo_root/scripts/validate-image-manifest.sh" "$manifest"
if [[ "$manifest" = /* ]]; then
  manifest_path="$(realpath "$manifest")"
else
  manifest_path="$(realpath "$repo_root/$manifest")"
fi

if [[ "$(jq -r '.build_enabled' "$manifest_path")" != "true" ]]; then
  echo "Image definition is disabled: $manifest" >&2
  jq -r '.disabled_reason' "$manifest_path" >&2
  exit 1
fi

platform="$(jq -r '.platform' "$manifest_path")"
context="$(jq -r '.context' "$manifest_path")"
dockerfile="$(jq -r '.dockerfile' "$manifest_path")"
repository="$(jq -r '.registry_repository' "$manifest_path")"
image_ref="$registry/$repository:$candidate_tag"

build_args=()
while IFS=$'\t' read -r key value; do
  [[ -n "$key" ]] || continue
  build_args+=(--build-arg "$key=$value")
done < <(jq -r '(.build_args // {}) | to_entries[] | [.key, .value] | @tsv' "$manifest_path")

docker buildx build \
  --platform "$platform" \
  --file "$repo_root/$dockerfile" \
  --tag "$image_ref" \
  --label "org.opencontainers.image.source=https://github.com/petabridge/netclaw-inference" \
  --label "org.opencontainers.image.revision=${GITHUB_SHA:-unknown}" \
  --label "org.opencontainers.image.version=$candidate_tag" \
  --provenance=mode=max \
  --sbom=true \
  --metadata-file "$metadata_file" \
  --push \
  "${build_args[@]}" \
  "$repo_root/$context"

digest="$(jq -r '.["containerimage.digest"] // empty' "$metadata_file")"
if [[ ! "$digest" =~ ^sha256:[a-f0-9]{64}$ ]]; then
  echo "BuildKit did not report a valid image digest in $metadata_file" >&2
  exit 1
fi

published_digest="$(docker buildx imagetools inspect "$image_ref" --format '{{json .Manifest}}' | jq -r '.digest')"
if [[ "$published_digest" != "$digest" ]]; then
  echo "Published digest mismatch: BuildKit=$digest registry=$published_digest" >&2
  exit 1
fi

echo "Published candidate: $image_ref@$digest"
