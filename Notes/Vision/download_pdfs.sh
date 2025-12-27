#!/usr/bin/env bash
set -euo pipefail

table_path="${1:-/home/jake/Developer/4D/Notes/Vision/Table-2.md}"
out_dir="${2:-/home/jake/Developer/4D/Notes/Vision/pdfs-3}"

if [[ ! -f "$table_path" ]]; then
  echo "Table not found: $table_path" >&2
  exit 1
fi

mkdir -p "$out_dir"

declare -A used_names=()
line_num=0

while IFS= read -r line; do
  line_num=$((line_num + 1))

  [[ "$line" =~ ^\|[[:space:]]*--- ]] && continue

  title=$(printf '%s' "$line" | awk -F'|' '{print $2}' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g; s/\*\*//g')
  pdf_cell=$(printf '%s' "$line" | awk -F'|' '{print $5}' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')

  [[ "$pdf_cell" == *"http"* ]] || continue

  url=$(printf '%s' "$pdf_cell" | sed -nE 's/.*\[[Pp][Dd][Ff]\]\((https?:\/\/[^)]+)\).*/\1/p')
  if [[ -z "$url" ]]; then
    url=$(printf '%s' "$pdf_cell" | sed -nE 's/.*(https?:\/\/[^ )`|]+).*/\1/p')
  fi

  if [[ -z "$url" ]]; then
    echo "Skipping line $line_num: no public PDF url found" >&2
    continue
  fi

  if [[ -z "$title" ]]; then
    title="paper_${line_num}"
  fi

  slug=$(printf '%s' "$title" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')
  if [[ -z "$slug" ]]; then
    slug="paper_${line_num}"
  fi

  filename="${slug}.pdf"
  if [[ -n "${used_names[$filename]+x}" ]]; then
    i=2
    while [[ -n "${used_names[${slug}_${i}.pdf]+x}" ]]; do
      i=$((i + 1))
    done
    filename="${slug}_${i}.pdf"
  fi
  used_names["$filename"]=1

  dest="${out_dir}/${filename}"
  if [[ -s "$dest" ]]; then
    echo "Skipping existing file: $dest"
    continue
  fi

  echo "Downloading: $title"
  curl -L --fail --retry 3 --retry-delay 1 -o "$dest" "$url"
done < "$table_path"

echo "Done. Files saved in: $out_dir"
