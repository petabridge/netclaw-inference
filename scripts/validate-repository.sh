#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

mapfile -t shell_files < <(find scripts -type f -name '*.sh' -print | sort)
for shell_file in "${shell_files[@]}"; do
  bash -n "$shell_file"
done

mapfile -t json_files < <(find . -path './.git' -prune -o -type f -name '*.json' -print | sort)
for json_file in "${json_files[@]}"; do
  jq empty "$json_file"
done

if unpinned_actions="$(
  grep -RInE '^[[:space:]]*-[[:space:]]+uses:[[:space:]]+[^./][^[:space:]]+@' .github/workflows 2>/dev/null \
    | grep -vE '@[a-f0-9]{40}([[:space:]]|$)' || true
)" && [[ -n "$unpinned_actions" ]]; then
  echo "GitHub Actions must be pinned to full commit SHAs:" >&2
  echo "$unpinned_actions" >&2
  exit 1
fi

mapfile -t manifests < <(find images -type f -name image.json -print | sort)
if [[ ${#manifests[@]} -eq 0 ]]; then
  echo "No image manifests found below images/." >&2
  exit 1
fi

ids_file="$(mktemp)"
repositories_file="$(mktemp)"
for manifest in "${manifests[@]}"; do
  "$repo_root/scripts/validate-image-manifest.sh" "$manifest"
  jq -r '.id' "$manifest" >> "$ids_file"
  jq -r '.registry_repository' "$manifest" >> "$repositories_file"
done

duplicate_ids="$(sort "$ids_file" | uniq -d)"
duplicate_repositories="$(sort "$repositories_file" | uniq -d)"
if [[ -n "$duplicate_ids" ]]; then
  echo "Duplicate image ids:" >&2
  echo "$duplicate_ids" >&2
  exit 1
fi
if [[ -n "$duplicate_repositories" ]]; then
  echo "Duplicate registry repositories:" >&2
  echo "$duplicate_repositories" >&2
  exit 1
fi

echo "Repository validation passed (${#manifests[@]} image manifest(s))."
