TARGET_FOLDER = "/home/jake/Developer/4D/Notes/Vision/pdfs-3"
IDENTIFIERS_PATH = "/home/jake/Developer/4D/Notes/Vision/Short-Identifiers.md"
FILE_EXT = ".pdf"
DRY_RUN = True
STRICT_MODE = False
ALLOW_FUZZY = False
FUZZY_CUTOFF = 0.92
FUZZY_MARGIN = 0.03
LOG_PATH = "rename_short_ids.log"

import argparse
import difflib
import os
import re
import sys
import time


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_event(log_path, message):
    line = f"{now_iso()} {message}\n"
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        sys.stderr.write(line)


def normalize(text):
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("×", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_identifiers(path):
    mapping = {}
    duplicates_norm = {}
    duplicates_short = {}
    short_to_title = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("|"):
                continue
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) < 2:
                continue
            short_id = parts[0]
            title = parts[1]
            if not short_id or not title:
                continue
            if short_id.lower().startswith("short identifier"):
                continue
            if set(short_id) == {"-"}:
                continue
            if short_id in short_to_title:
                duplicates_short.setdefault(short_id, [short_to_title[short_id]]).append(
                    title
                )
            else:
                short_to_title[short_id] = title
            norm_title = normalize(title)
            if norm_title in mapping:
                existing = mapping[norm_title]
                if existing:
                    duplicates_norm.setdefault(norm_title, [existing]).append(short_id)
                else:
                    duplicates_norm.setdefault(norm_title, []).append(short_id)
                mapping[norm_title] = None
            else:
                mapping[norm_title] = short_id
    return mapping, duplicates_norm, duplicates_short


def list_pdfs(folder, file_ext):
    entries = []
    for name in os.listdir(folder):
        if not name.lower().endswith(file_ext.lower()):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            entries.append(path)
    return sorted(entries)


def best_fuzzy_match(norm_name, title_keys):
    best_key = None
    best_ratio = 0.0
    second_ratio = 0.0
    for key in title_keys:
        ratio = difflib.SequenceMatcher(None, norm_name, key).ratio()
        if ratio > best_ratio:
            second_ratio = best_ratio
            best_ratio = ratio
            best_key = key
        elif ratio > second_ratio:
            second_ratio = ratio
    return best_key, best_ratio, second_ratio


def match_title(norm_name, mapping, allow_fuzzy):
    if norm_name in mapping:
        if mapping[norm_name] is None:
            return None, "ambiguous", 0.0
        return mapping[norm_name], "exact", 1.0
    if not allow_fuzzy:
        return None, "unmatched", 0.0
    best_key, best_ratio, second_ratio = best_fuzzy_match(
        norm_name, mapping.keys()
    )
    if best_key is None:
        return None, "unmatched", 0.0
    if best_ratio < FUZZY_CUTOFF:
        return None, "unmatched", best_ratio
    if (best_ratio - second_ratio) < FUZZY_MARGIN:
        return None, "ambiguous", best_ratio
    return mapping[best_key], "fuzzy", best_ratio


def build_plan(pdf_paths, mapping, allow_fuzzy, file_ext):
    plan = []
    for path in pdf_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        norm_name = normalize(stem)
        short_id, match_type, score = match_title(
            norm_name, mapping, allow_fuzzy
        )
        if not short_id:
            plan.append(
                {
                    "source": path,
                    "target": None,
                    "status": "unmatched" if match_type == "unmatched" else match_type,
                    "score": score,
                    "short_id": None,
                }
            )
            continue
        target = os.path.join(os.path.dirname(path), short_id + file_ext)
        plan.append(
            {
                "source": path,
                "target": target,
                "status": "ok",
                "score": score,
                "short_id": short_id,
                "match_type": match_type,
            }
        )
    return plan


def detect_conflicts(plan):
    errors = []
    target_map = {}
    for item in plan:
        if item["status"] != "ok":
            continue
        target = item["target"]
        if target in target_map:
            other = target_map[target]
            errors.append(
                f"duplicate_target target={target} sources={other['source']} and {item['source']}"
            )
            other["status"] = "conflict"
            item["status"] = "conflict"
            continue
        target_map[target] = item
        if os.path.exists(target) and os.path.abspath(target) != os.path.abspath(
            item["source"]
        ):
            errors.append(f"target_exists target={target} source={item['source']}")
            item["status"] = "conflict"
    return errors


def print_plan(plan, dry_run):
    mode = "DRY RUN" if dry_run else "APPLY"
    print(f"{mode}: rename plan")
    for item in plan:
        src_name = os.path.basename(item["source"])
        if item["status"] == "ok":
            dst_name = os.path.basename(item["target"])
            match_type = item.get("match_type", "exact")
            print(f"  {src_name} -> {dst_name} ({match_type})")
        else:
            print(f"  {src_name} -> skip ({item['status']})")


def apply_plan(plan, log_path):
    renamed = 0
    skipped = 0
    for item in plan:
        if item["status"] != "ok":
            skipped += 1
            continue
        source = item["source"]
        target = item["target"]
        if os.path.abspath(source) == os.path.abspath(target):
            skipped += 1
            log_event(log_path, f"already_named path={source}")
            continue
        try:
            os.rename(source, target)
            renamed += 1
            log_event(log_path, f"renamed source={source} target={target}")
        except Exception as exc:
            skipped += 1
            log_event(
                log_path, f"rename_failed source={source} target={target} error={exc}"
            )
    return renamed, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename PDFs to short identifiers."
    )
    parser.add_argument("--folder", default=TARGET_FOLDER, help="PDF folder.")
    parser.add_argument(
        "--map-path",
        default=IDENTIFIERS_PATH,
        help="Path to Short-Identifiers.md.",
    )
    dry_group = parser.add_mutually_exclusive_group()
    dry_group.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="Apply renames.",
    )
    dry_group.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="Show the plan without renaming.",
    )
    parser.add_argument(
        "--allow-fuzzy",
        action="store_true",
        help="Allow fuzzy matching if exact title match fails.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Abort if any file is unmatched or ambiguous.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow partial renames even if some files are unmatched.",
    )
    parser.set_defaults(strict=None, apply=None)
    return parser.parse_args()


def main():
    args = parse_args()
    folder = os.path.abspath(args.folder)
    map_path = os.path.abspath(args.map_path)
    dry_run = DRY_RUN if args.apply is None else not args.apply
    allow_fuzzy = ALLOW_FUZZY or args.allow_fuzzy
    strict_mode = STRICT_MODE if args.strict is None else args.strict

    log_path = LOG_PATH
    if not os.path.isabs(log_path):
        log_path = os.path.abspath(os.path.join(folder, log_path))

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return 1
    if not os.path.exists(map_path):
        print(f"Short-Identifiers file not found: {map_path}")
        return 1

    log_event(
        log_path,
        f"run_start folder={folder} dry_run={dry_run} strict={strict_mode} allow_fuzzy={allow_fuzzy}",
    )

    mapping, duplicates_norm, duplicates_short = parse_identifiers(map_path)
    if duplicates_norm:
        for norm_title, short_ids in duplicates_norm.items():
            log_event(
                log_path,
                f"duplicate_title norm={norm_title} short_ids={short_ids}",
            )
        if strict_mode:
            print("Duplicate titles in Short-Identifiers.md; aborting.")
            return 2
    if duplicates_short:
        for short_id, titles in duplicates_short.items():
            log_event(
                log_path,
                f"duplicate_short_id short_id={short_id} titles={titles}",
            )
        if strict_mode:
            print("Duplicate short identifiers found; aborting.")
            return 2

    pdf_paths = list_pdfs(folder, FILE_EXT)
    if not pdf_paths:
        print("No PDFs found.")
        return 0

    plan = build_plan(pdf_paths, mapping, allow_fuzzy, FILE_EXT)
    conflict_errors = detect_conflicts(plan)
    if conflict_errors:
        for err in conflict_errors:
            log_event(log_path, f"conflict {err}")
        if strict_mode:
            print("Conflicts detected; aborting.")
            print("See log for details.")
            return 2

    unmatched = [item for item in plan if item["status"] != "ok"]
    if unmatched:
        for item in unmatched:
            log_event(
                log_path,
                f"unmatched source={item['source']} status={item['status']} score={item['score']}",
            )
        if strict_mode:
            print("Unmatched or ambiguous files detected; aborting.")
            print("See log for details.")
            return 2

    print_plan(plan, dry_run)
    ok_count = sum(1 for item in plan if item["status"] == "ok")
    skip_count = len(plan) - ok_count
    if skip_count:
        print(f"Skipping {skip_count} file(s) due to unmatched/ambiguous/conflict.")

    if dry_run:
        log_event(
            log_path, f"dry_run_complete ok={ok_count} skipped={skip_count}"
        )
        return 0

    renamed, skipped = apply_plan(plan, log_path)
    log_event(log_path, f"run_end renamed={renamed} skipped={skipped}")
    print(f"Renamed: {renamed}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
