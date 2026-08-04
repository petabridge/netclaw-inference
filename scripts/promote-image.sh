#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 <path-to-image.json> <sha256:digest> <vMAJOR.MINOR.PATCH[-suffix]>" >&2
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
manifest="$1"
digest="$2"
release_tag="$3"
registry="${REGISTRY:-docker.testlab.petabridge.net}"

if [[ ! "$digest" =~ ^sha256:[a-f0-9]{64}$ ]]; then
  echo "Promotion requires an exact sha256 image digest." >&2
  exit 1
fi
if [[ ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-z0-9][a-z0-9.-]*)?$ ]]; then
  echo "Release tags must use semantic version syntax such as v0.1.0 or v0.1.0-rc.1." >&2
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
  echo "Image definition is disabled and cannot be promoted: $manifest" >&2
  exit 1
fi
repository="$(jq -r '.registry_repository' "$manifest_path")"
source_ref="$registry/$repository@$digest"
destination_ref="$registry/$repository:$release_tag"

source_digest="$(docker buildx imagetools inspect "$source_ref" --format '{{json .Manifest}}' | jq -r '.digest')"
if [[ "$source_digest" != "$digest" ]]; then
  echo "Source digest verification failed: expected=$digest actual=$source_digest" >&2
  exit 1
fi

if existing_json="$(docker buildx imagetools inspect "$destination_ref" --format '{{json .Manifest}}' 2>/dev/null)"; then
  existing_digest="$(jq -r '.digest' <<< "$existing_json")"
  if [[ "$existing_digest" == "$digest" ]]; then
    echo "Release is already promoted: $destination_ref@$digest"
    exit 0
  fi
  echo "Refusing to replace existing release tag $destination_ref ($existing_digest)." >&2
  exit 1
fi

docker buildx imagetools create --tag "$destination_ref" "$source_ref"
promoted_digest="$(docker buildx imagetools inspect "$destination_ref" --format '{{json .Manifest}}' | jq -r '.digest')"
if [[ "$promoted_digest" != "$digest" ]]; then
  echo "Promotion changed the manifest digest: expected=$digest actual=$promoted_digest" >&2
  exit 1
fi

echo "Promoted release: $destination_ref@$digest"
