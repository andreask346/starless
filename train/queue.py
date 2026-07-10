# Unattended experiment queue for the away-days.
#   .venv\Scripts\python.exe train\queue.py
# Runs experiments from train\queue.json sequentially:
#   train (subprocess, auto-resume, crash-retry) -> export ONNX -> eval panel
# After every experiment (and every heartbeat) writes STATUS.md and, if a git
# remote exists, commits+pushes it so Andreas can check progress from his
# phone on GitHub.
#
# Kill switch: create a file named STOP in the repo root — the queue finishes
# the current checkpoint interval and exits cleanly. Everything is resumable:
# re-running queue.py continues where it left off.

import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
QUEUE = os.path.join(ROOT, "train", "queue.json")
STATUS = os.path.join(ROOT, "STATUS.md")
STOP = os.path.join(ROOT, "STOP")
MAX_RETRIES = 5


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def push_status(lines):
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(f"# Starless training status\n\nupdated {now()}\n\n"
                + "\n".join(lines) + "\n")
    try:
        subprocess.run(["git", "add", "STATUS.md"], cwd=ROOT, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"status {now()}"],
                       cwd=ROOT, capture_output=True)
        r = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True,
                           timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def tail_log(run_name, n=3):
    p = os.path.join(ROOT, "runs", run_name, "log.jsonl")
    if not os.path.isfile(p):
        return []
    with open(p) as f:
        return [ln.strip() for ln in f.readlines()[-n:]]


def run_experiment(exp, done, all_lines):
    name = exp["name"]
    args = [PY, os.path.join(ROOT, "train", "train.py"), "--name", name,
            "--resume"]
    for k, v in exp.get("args", {}).items():
        args += [f"--{k}", str(v)] if v is not True else [f"--{k}"]
    retries = 0
    while True:
        if os.path.exists(STOP):
            return "stopped"
        started = now()
        all_lines_hdr = [f"## RUNNING: {name} (attempt {retries + 1}, "
                         f"started {started})"]
        push_status(done + all_lines_hdr + tail_log(name))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(args, cwd=ROOT, env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
        # heartbeat loop: status push every 30 min while training
        while proc.poll() is None:
            for _ in range(180):                    # 30 min in 10s slices
                time.sleep(10)
                if proc.poll() is not None:
                    break
                if os.path.exists(STOP):
                    proc.terminate()
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return "stopped"
            push_status(done + [f"## RUNNING: {name} ({now()})"]
                        + tail_log(name))
        if proc.returncode == 0:
            return "ok"
        retries += 1
        if retries >= MAX_RETRIES:
            return f"FAILED after {MAX_RETRIES} attempts"
        time.sleep(60)                              # cool off, then resume


def post_steps(exp, lines):
    name = exp["name"]
    best = os.path.join(ROOT, "runs", name, "best.pt")
    onnx = os.path.join(ROOT, "runs", name, f"{name}.onnx")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r1 = subprocess.run([PY, os.path.join(ROOT, "export", "export_onnx.py"),
                         best, onnx], cwd=ROOT, env=env,
                        capture_output=True, text=True, timeout=1800)
    lines.append(f"- export: {'ok' if r1.returncode == 0 else 'FAILED'}")
    r2 = subprocess.run([PY, os.path.join(ROOT, "eval", "panel.py"), best],
                        cwd=ROOT, env=env, capture_output=True, text=True,
                        timeout=7200)
    metric_line = next((ln for ln in r2.stdout.splitlines()
                        if ln.startswith("METRICS ")), "")
    lines.append(f"- eval: {metric_line[8:300] if metric_line else 'FAILED'}")


def main():
    exps = json.load(open(QUEUE))
    done = [f"queue of {len(exps)} experiments"]
    for i, exp in enumerate(exps):
        if os.path.exists(STOP):
            done.append(f"STOPPED by kill switch before {exp['name']}")
            break
        marker = os.path.join(ROOT, "runs", exp["name"], "DONE")
        if os.path.isfile(marker):
            done.append(f"## {exp['name']}: done earlier (skipped)")
            continue
        t0 = time.time()
        result = run_experiment(exp, done, [])
        hours = (time.time() - t0) / 3600
        lines = [f"## {exp['name']}: {result} ({hours:.1f} h)"]
        if result == "ok":
            post_steps(exp, lines)
            open(marker, "w").write(now())
        done += lines
        push_status(done + [f"-- next: {exps[i+1]['name']}"
                            if i + 1 < len(exps) else "-- queue complete"])
        if result == "stopped":
            break
    push_status(done + ["", f"queue exited {now()}"])
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
