#!/bin/bash
set -euo pipefail

# Assert that the root README Images table is an accurate inventory of the
# image manifests in this repository. Every manifest must be listed exactly
# once, linked to the README.md that sits next to its image.json, and no
# listed row may point at an image that no longer has a manifest. A stale
# table is a public wrong answer, so validate.yml blocks the merge.

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
readme="$repo_root/README.md"

if [[ ! -f "$readme" ]]; then
  echo "Missing README.md at repository root." >&2
  exit 1
fi

# Pull the "| [`<id>`](<path>) | ..." rows out of the "## Images" section only,
# so unrelated tables cannot satisfy or break coverage. Image ids and
# repository-relative paths contain no backticks, so dropping every backtick
# normalizes the link text to the bare id.
readme_row_pairs() {
  awk '
    /^## Images[[:space:]]*$/ { in_section = 1; next }
    in_section && /^## / { exit }
    in_section && /^\|[[:space:]]*\[[^]]*\]\([^)]*\)/ {
      line = $0
      sub(/^\|[[:space:]]*\[/, "", line)
      sep = index(line, "](")
      id = substr(line, 1, sep - 1)
      rest = substr(line, sep + 2)
      sub(/\).*$/, "", rest)
      gsub(/`/, "", id)
      printf "%s\t%s\n", id, rest
    }
  ' "$1"
}

# Emit "<id>\t<repo-relative path to the image's own README.md>" per manifest.
manifest_pairs() {
  local manifest relative
  while IFS= read -r manifest; do
    relative="${manifest#"$repo_root"/}"
    printf '%s\t%s\n' "$(jq -r '.id' "$manifest")" "$(dirname "$relative")/README.md"
  done < <(find "$repo_root/images" -type f -name image.json -print | sort)
}

expected_file="$(mktemp)"
listed_file="$(mktemp)"
trap 'rm -f "$expected_file" "$listed_file"' EXIT

manifest_pairs > "$expected_file"
readme_row_pairs "$readme" > "$listed_file"

if [[ ! -s "$expected_file" ]]; then
  echo "No image manifests found below images/." >&2
  exit 1
fi

failures=0
missing=0

# Forward direction: every manifest is listed once, linked to the right file,
# and that file exists.
while IFS=$'\t' read -r id path; do
  if [[ ! -f "$repo_root/$path" ]]; then
    echo "Image directory has no README to link from the Images table: $path" >&2
    failures=$((failures + 1))
  fi

  row_count="$(awk -F'\t' -v want="$id" '$1 == want { n++ } END { print n + 0 }' "$listed_file")"
  if [[ "$row_count" -eq 0 ]]; then
    echo "Image id is missing from the README Images table: $id" >&2
    missing=$((missing + 1))
    failures=$((failures + 1))
    continue
  fi
  if [[ "$row_count" -gt 1 ]]; then
    echo "Image id is listed ${row_count} times in the README Images table: $id" >&2
    failures=$((failures + 1))
    continue
  fi

  listed_path="$(awk -F'\t' -v want="$id" '$1 == want { print $2; exit }' "$listed_file")"
  if [[ "$listed_path" != "$path" ]]; then
    echo "README Images row for $id links $listed_path; expected $path" >&2
    failures=$((failures + 1))
  fi
done < "$expected_file"

# Reverse direction: no row references an image that has no manifest.
while IFS=$'\t' read -r id path; do
  if ! awk -F'\t' -v want="$id" 'BEGIN { found = 0 } $1 == want { found = 1 } END { exit !found }' \
    "$expected_file"; then
    echo "README Images table lists unknown image id: $id (links $path)" >&2
    failures=$((failures + 1))
  fi
done < "$listed_file"

if [[ "$failures" -ne 0 ]]; then
  echo "README image coverage failed with ${failures} problem(s)." >&2
  if [[ "$missing" -ne 0 ]]; then
    echo "Add an Images-table row for each missing id, linked to its image README.md." >&2
  fi
  exit 1
fi

manifest_count="$(wc -l < "$expected_file" | tr -d '[:space:]')"
echo "Validated README image coverage (${manifest_count} image(s) listed)."
