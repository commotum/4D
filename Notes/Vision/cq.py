TARGET_FOLDER = "."
SURVEY_MD_PATH = "/home/jake/Developer/4D/Notes/Vision/Survey.md"
CODEX_CLI_CMD = "codex"
FILE_EXT = ".pdf"
DRY_RUN = False
OVERWRITE_MD = False
SORT_MODE = "alpha"
LOG_PATH = "codex_queue.log"
CODEX_EXEC_ARGS = ["exec", "--full-auto"]  # Adjust Codex CLI mode/permissions here.
CODEX_EXEC_TIMEOUT = 3600
CODEX_CWD = None
CODEX_ADD_DIR = True
CODEX_SKIP_GIT_CHECK = False
MAX_LOG_CHARS = 2000
MIN_MD_BYTES = 1

import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

PROMPT_TEMPLATE = (
    "1. Read and review [PDF_ABS_PATH] thoroughly. Make sure the files entire contents has been brought into the context. This will be absolutely essential.\n"
    "2. For each question outlined in [SURVEY_MD_PATH] determine the answer. Double check and verify your answer.\n"
    "3. Copy your answer for the given question into [MD_ABS_PATH] be absolutely thorough. Save it.\n"
    "4. Continue until you have answered all the questions\n"
)


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


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


def load_state(state_path, log_path):
    if not os.path.exists(state_path):
        return {"completed": {}}
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("state is not a dict")
        if "completed" not in data or not isinstance(data["completed"], dict):
            data["completed"] = {}
        return data
    except Exception as exc:
        log_event(log_path, f"state_load_failed path={state_path} error={exc}")
        return {"completed": {}}


def save_state(state_path, state, log_path):
    try:
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
    except Exception as exc:
        log_event(log_path, f"state_save_failed path={state_path} error={exc}")


def find_git_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def sanitize_log_text(text, max_chars):
    if not text:
        return ""
    scrubbed = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(scrubbed) <= max_chars:
        return scrubbed
    return scrubbed[:max_chars] + "...(truncated)"


def quote_path(path):
    if any(ch.isspace() for ch in path):
        return '"' + path.replace('"', '\\"') + '"'
    return path


def build_prompt(pdf_abs_path, md_abs_path, survey_md_path):
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("[PDF_ABS_PATH]", quote_path(pdf_abs_path))
    prompt = prompt.replace("[SURVEY_MD_PATH]", quote_path(survey_md_path))
    prompt = prompt.replace("[MD_ABS_PATH]", quote_path(md_abs_path))
    return prompt


def list_pdf_files(target_folder, file_ext):
    entries = []
    for name in os.listdir(target_folder):
        if not name.lower().endswith(file_ext.lower()):
            continue
        full_path = os.path.join(target_folder, name)
        if os.path.isfile(full_path):
            entries.append(name)
    return entries


def sort_queue(entries, target_folder, sort_mode, log_path):
    mode = (sort_mode or "").strip().lower()
    if mode == "alpha":
        return sorted(entries, key=lambda n: n.lower())
    if mode in ("mtime", "mtime-desc", "newest"):
        return sorted(
            entries,
            key=lambda n: os.path.getmtime(os.path.join(target_folder, n)),
            reverse=True,
        )
    if mode in ("mtime-asc", "oldest"):
        return sorted(
            entries,
            key=lambda n: os.path.getmtime(os.path.join(target_folder, n)),
        )
    log_event(log_path, f"unknown_sort_mode mode={sort_mode} fallback=alpha")
    return sorted(entries, key=lambda n: n.lower())


def ensure_md_file(md_abs_path, overwrite, dry_run, log_path):
    if dry_run:
        log_event(
            log_path,
            f"dry_run_md_prepare path={md_abs_path} overwrite={overwrite}",
        )
        return True
    try:
        with open(md_abs_path, "w", encoding="utf-8") as handle:
            handle.write("")
        return True
    except Exception as exc:
        log_event(log_path, f"md_prepare_failed path={md_abs_path} error={exc}")
        return False


def build_codex_exec_command(codex_cmd, codex_cwd, add_dir, skip_git_check):
    args = shlex.split(codex_cmd)
    if not args:
        return []
    if "exec" not in args:
        args.extend(CODEX_EXEC_ARGS)
    if codex_cwd and "-C" not in args and "--cd" not in args:
        args.extend(["-C", codex_cwd])
    if add_dir and "--add-dir" not in args:
        args.extend(["--add-dir", add_dir])
    if skip_git_check and "--skip-git-repo-check" not in args:
        args.append("--skip-git-repo-check")
    return args


def run_codex_exec(codex_cmd, prompt, codex_cwd, add_dir, skip_git_check, log_path):
    args = build_codex_exec_command(codex_cmd, codex_cwd, add_dir, skip_git_check)
    if not args:
        return False, "empty_codex_cmd"
    base_cmd = args[0]
    if shutil.which(base_cmd) is None:
        log_event(log_path, f"codex_missing cmd={base_cmd}")
        return False, "codex_missing"
    cmd_display = " ".join(shlex.quote(arg) for arg in args)
    log_event(log_path, f"codex_launch cmd={cmd_display}")
    start_time = time.monotonic()
    try:
        result = subprocess.run(
            args,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=CODEX_EXEC_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start_time
        log_event(log_path, f"codex_timeout seconds={elapsed:.1f}")
        return False, "timeout"
    except Exception as exc:
        log_event(log_path, f"codex_exception error={exc}")
        return False, str(exc)
    elapsed = time.monotonic() - start_time
    if result.returncode != 0:
        stderr_trimmed = sanitize_log_text(result.stderr, MAX_LOG_CHARS)
        stdout_trimmed = sanitize_log_text(result.stdout, MAX_LOG_CHARS)
        log_event(
            log_path,
            f"codex_failure rc={result.returncode} seconds={elapsed:.1f} stderr={stderr_trimmed} stdout={stdout_trimmed}",
        )
        return False, f"rc={result.returncode}"
    log_event(log_path, f"codex_success seconds={elapsed:.1f}")
    return True, None


def md_has_content(md_abs_path):
    try:
        return os.path.getsize(md_abs_path) >= MIN_MD_BYTES
    except OSError:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run Codex CLI for a queue of PDFs.")
    parser.add_argument("--folder", default=TARGET_FOLDER, help="Folder with PDFs.")
    parser.add_argument(
        "--codex-cmd",
        default=CODEX_CLI_CMD,
        help="Command to launch Codex CLI.",
    )
    parser.add_argument(
        "--sort-mode",
        default=None,
        help="Sort mode: alpha, mtime, mtime-asc, mtime-desc, newest, oldest.",
    )
    dry_group = parser.add_mutually_exclusive_group()
    dry_group.add_argument("--dry-run", dest="dry_run", action="store_true")
    dry_group.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=None)
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", dest="overwrite", action="store_true")
    overwrite_group.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=None)
    return parser.parse_args()


def main():
    args = parse_args()
    target_folder = os.path.abspath(args.folder)
    if not os.path.isdir(target_folder):
        print(f"Target folder not found: {target_folder}")
        return 1

    dry_run = DRY_RUN if args.dry_run is None else args.dry_run
    overwrite = OVERWRITE_MD if args.overwrite is None else args.overwrite
    sort_mode = SORT_MODE if args.sort_mode is None else args.sort_mode
    codex_cmd = args.codex_cmd

    log_path = LOG_PATH
    if not os.path.isabs(log_path):
        log_path = os.path.abspath(os.path.join(target_folder, log_path))
    state_path = os.path.join(target_folder, ".codex_queue_state.json")

    log_event(
        log_path,
        f"run_start folder={target_folder} dry_run={dry_run} overwrite={overwrite} sort_mode={sort_mode}",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    codex_cwd = os.path.abspath(CODEX_CWD) if CODEX_CWD else None
    if codex_cwd is None:
        codex_cwd = find_git_root(script_dir) or find_git_root(target_folder)
    if codex_cwd:
        log_event(log_path, f"codex_cwd path={codex_cwd}")
    else:
        log_event(log_path, "codex_cwd_missing")
    add_dir = target_folder if CODEX_ADD_DIR else None
    skip_git_check = CODEX_SKIP_GIT_CHECK

    survey_md_path = os.path.abspath(SURVEY_MD_PATH)
    if not os.path.exists(survey_md_path):
        log_event(log_path, f"survey_missing path={survey_md_path}")

    state = load_state(state_path, log_path)
    completed = state.get("completed", {})

    entries = list_pdf_files(target_folder, FILE_EXT)
    queue = sort_queue(entries, target_folder, sort_mode, log_path)

    total = len(queue)
    if total == 0:
        print("No PDF files found.")
        log_event(log_path, "no_pdfs_found")
        return 0

    for index, name in enumerate(queue, start=1):
        pdf_abs_path = os.path.abspath(os.path.join(target_folder, name))
        md_abs_path = os.path.splitext(pdf_abs_path)[0] + ".md"
        print(f"[{index}/{total}] processing {name}")
        log_event(log_path, f"file_start pdf={pdf_abs_path} md={md_abs_path}")

        if not overwrite and pdf_abs_path in completed:
            log_event(log_path, f"skipped_state pdf={pdf_abs_path}")
            continue

        if not overwrite and os.path.exists(md_abs_path):
            log_event(log_path, f"skipped_existing_md pdf={pdf_abs_path}")
            if not dry_run:
                completed[pdf_abs_path] = now_iso()
                state["completed"] = completed
                save_state(state_path, state, log_path)
            continue

        if not ensure_md_file(md_abs_path, overwrite, dry_run, log_path):
            log_event(log_path, f"skipped_md_prepare_failed pdf={pdf_abs_path}")
            continue

        prompt = build_prompt(pdf_abs_path, md_abs_path, survey_md_path)
        log_event(log_path, f"prompt_generated pdf={pdf_abs_path}")

        if dry_run:
            print("DRY RUN prompt:")
            print(prompt)
            log_event(log_path, f"dry_run_skip pdf={pdf_abs_path}")
            continue

        # Adjust CODEX_EXEC_ARGS above to change how Codex runs (e.g., --full-auto).
        success, error = run_codex_exec(
            codex_cmd,
            prompt,
            codex_cwd,
            add_dir,
            skip_git_check,
            log_path,
        )
        if success:
            if md_has_content(md_abs_path):
                size = os.path.getsize(md_abs_path)
                log_event(log_path, f"md_written pdf={pdf_abs_path} bytes={size}")
                completed[pdf_abs_path] = now_iso()
                state["completed"] = completed
                save_state(state_path, state, log_path)
            else:
                log_event(log_path, f"md_empty pdf={pdf_abs_path}")
                print(f"  warning: output file is empty for {name} (see log)")
        else:
            log_event(log_path, f"codex_failure pdf={pdf_abs_path} error={error}")
            if error == "codex_missing":
                print("Codex CLI not found. Paste this prompt into Codex manually:")
                print(prompt)
                log_event(log_path, f"manual_prompt_printed pdf={pdf_abs_path}")

    log_event(log_path, "run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
