TARGET_FOLDER = "."
SURVEY_MD_PATH = "/home/jake/Developer/4D/Notes/Vision/Survey.md"
CODEX_CLI_CMD = "codex"
FILE_EXT = ".pdf"
DRY_RUN = False
OVERWRITE_MD = False
SORT_MODE = "alpha"
LOG_PATH = "codex_queue.log"
EXIT_COMMANDS = ["/quit"]
PEXPECT_START_DELAY = 1.0
PEXPECT_TIMEOUT = 45.0
PEXPECT_EXIT_TIMEOUT = 15.0

import argparse
import datetime
import json
import os
import shlex
import sys
import time

try:
    import pexpect

    HAVE_PEXPECT = True
except Exception:
    HAVE_PEXPECT = False

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


def run_codex_with_prompt(codex_cmd, prompt, exit_commands, log_path):
    if not HAVE_PEXPECT:
        return False, "pexpect_missing"
    args = shlex.split(codex_cmd)
    if not args:
        return False, "empty_codex_cmd"
    child = None
    try:
        child = pexpect.spawn(
            args[0],
            args[1:],
            encoding="utf-8",
            timeout=PEXPECT_TIMEOUT,
        )
        time.sleep(PEXPECT_START_DELAY)
        # Adjust this block if Codex CLI needs a different prompt injection workflow.
        child.sendline(prompt)
        # Adjust EXIT_COMMANDS above if Codex uses a different quit command.
        for exit_cmd in exit_commands:
            child.sendline(exit_cmd)
        child.expect(pexpect.EOF, timeout=PEXPECT_EXIT_TIMEOUT)
        child.close()
        return True, None
    except pexpect.exceptions.TIMEOUT as exc:
        if child is not None:
            child.terminate(force=True)
        log_event(log_path, f"codex_timeout error={exc}")
        return False, "timeout"
    except Exception as exc:
        if child is not None:
            try:
                child.terminate(force=True)
            except Exception:
                pass
        log_event(log_path, f"codex_exception error={exc}")
        return False, str(exc)


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

    if not HAVE_PEXPECT and not dry_run:
        log_event(log_path, "pexpect_unavailable_manual_mode")

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

        if not HAVE_PEXPECT:
            print("Manual mode (pexpect not available). Paste this prompt into Codex:")
            print(prompt)
            log_event(log_path, f"manual_prompt_printed pdf={pdf_abs_path}")
            continue

        log_event(log_path, f"codex_launch pdf={pdf_abs_path}")
        success, error = run_codex_with_prompt(
            codex_cmd,
            prompt,
            EXIT_COMMANDS,
            log_path,
        )
        if success:
            log_event(log_path, f"codex_success pdf={pdf_abs_path}")
            completed[pdf_abs_path] = now_iso()
            state["completed"] = completed
            save_state(state_path, state, log_path)
        else:
            log_event(log_path, f"codex_failure pdf={pdf_abs_path} error={error}")

    log_event(log_path, "run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
