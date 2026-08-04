#!/bin/bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  validate-release-notes.sh
  validate-release-notes.sh --changed-since <base-commit>
  validate-release-notes.sh --release <vMAJOR.MINOR.PATCH[-suffix]> <image-id>
USAGE
}

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
notes="$repo_root/RELEASE_NOTES.md"

if [[ ! -f "$notes" ]]; then
  echo "Missing RELEASE_NOTES.md" >&2
  exit 1
fi

mapfile -t headers < <(grep -E '^## v[0-9]+\.[0-9]+\.[0-9]+(-[a-z0-9][a-z0-9.-]*)? \((Unreleased|[0-9]{4}-[0-9]{2}-[0-9]{2})\)$' "$notes")
if [[ ${#headers[@]} -eq 0 ]]; then
  echo "RELEASE_NOTES.md has no valid release header." >&2
  exit 1
fi

all_header_count="$(grep -c '^## ' "$notes" || true)"
if [[ "$all_header_count" -ne "${#headers[@]}" ]]; then
  echo "Every level-two release header must use: ## vX.Y.Z (Unreleased|YYYY-MM-DD)" >&2
  exit 1
fi

versions="$(printf '%s\n' "${headers[@]}" | sed -E 's/^## (v[^ ]+) .*/\1/')"
duplicates="$(printf '%s\n' "$versions" | sort | uniq -d)"
if [[ -n "$duplicates" ]]; then
  echo "Duplicate release-note versions:" >&2
  echo "$duplicates" >&2
  exit 1
fi

unreleased_count="$(grep -cE '^## v[^ ]+ \(Unreleased\)$' "$notes" || true)"
if [[ "$unreleased_count" -gt 1 ]]; then
  echo "RELEASE_NOTES.md may contain at most one Unreleased section." >&2
  exit 1
fi
if [[ "$unreleased_count" -eq 1 && "${headers[0]}" != *"(Unreleased)" ]]; then
  echo "The Unreleased section must be the first release section." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Release-note structure is valid (${#headers[@]} release(s))."
  exit 0
fi

case "$1" in
  --changed-since)
    if [[ $# -ne 2 ]]; then
      usage
      exit 2
    fi
    base_commit="$2"
    if ! git -C "$repo_root" cat-file -e "$base_commit^{commit}" 2>/dev/null; then
      echo "Base commit is unavailable: $base_commit" >&2
      exit 1
    fi

    mapfile -t changed_files < <(git -C "$repo_root" diff --name-only "$base_commit"...HEAD)
    build_change=false
    notes_changed=false
    for changed_file in "${changed_files[@]}"; do
      if [[ "$changed_file" == "RELEASE_NOTES.md" ]]; then
        notes_changed=true
      fi
      case "$changed_file" in
        images/*/Dockerfile|images/*/Dockerfile.*|images/*/image.json|images/*/*.lock|images/*/patches/*|images/*/overlay/*|scripts/build-image.sh|scripts/promote-image.sh)
          build_change=true
          ;;
      esac
    done

    if [[ "$build_change" == "true" && "$notes_changed" != "true" ]]; then
      echo "Build-affecting changes require a RELEASE_NOTES.md update." >&2
      exit 1
    fi
    echo "Release-note change policy passed."
    ;;

  --release)
    if [[ $# -ne 3 ]]; then
      usage
      exit 2
    fi
    release_tag="$2"
    image_id="$3"
    if [[ ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-z0-9][a-z0-9.-]*)?$ ]]; then
      echo "Invalid release tag: $release_tag" >&2
      exit 1
    fi
    if [[ ! "$image_id" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
      echo "Invalid image id: $image_id" >&2
      exit 1
    fi

    section="$(awk -v header="## $release_tag " '
      index($0, header) == 1 { found=1 }
      found && /^## / && index($0, header) != 1 { exit }
      found { print }
    ' "$notes")"
    if [[ -z "$section" ]]; then
      echo "No release-note section exists for $release_tag." >&2
      exit 1
    fi
    if ! grep -qE "^## ${release_tag//./\\.} \([0-9]{4}-[0-9]{2}-[0-9]{2}\)$" <<< "$section"; then
      echo "Release $release_tag must have a date instead of Unreleased." >&2
      exit 1
    fi
    if ! grep -Fq -- "- \`$image_id\`" <<< "$section"; then
      echo "Release $release_tag does not list image id $image_id." >&2
      exit 1
    fi
    echo "Release notes authorize $image_id at $release_tag."
    ;;

  *)
    usage
    exit 2
    ;;
esac
