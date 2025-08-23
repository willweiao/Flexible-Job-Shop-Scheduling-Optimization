"""
viz_cli.py — Static charts for FJSP results (Matplotlib version)
Usage:
    python scripts/viz_cli.py --case mk01
or:
    python scripts/viz_cli.py  # then input index, e.g., 1 -> mk01

Inputs (required in outputs/<case>/):
    - schedule_gantt.csv  with columns:
      machine,job,op,start,finish,duration,op_id

Outputs (in outputs/<case>/figs/):
    - gantt_<case>.png
    - util_<case>.png
    - job_completion_<case>.png
    - wip_<case>.png
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

# ---------- Data ----------
def load_schedule(case: str) -> pd.DataFrame:
    csv_path = Path("outputs") / case / "schedule_gantt.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")
    df = pd.read_csv(csv_path)
    need = {"machine","job","op","start","finish","duration","op_id"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in schedule_gantt.csv: {sorted(miss)}")
    # types
    df["machine"] = df["machine"].astype(str)
    df["job"] = df["job"].astype(str)
    df["start"] = df["start"].astype(float)
    df["finish"] = df["finish"].astype(float)
    df["duration"] = df["duration"].astype(float)
    return df

# ---------- Figures ----------
def plot_gantt(df: pd.DataFrame, case: str, path: Path):
    """Machine-wise Gantt using broken_barh"""
    df = df.sort_values(["machine", "start", "finish"]).copy()
    machines = sorted(df["machine"].unique())
    # map machine to row index
    m2row = {m:i for i,m in enumerate(machines)}
    fig_h = max(3.5, 0.5*len(machines) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # choose a simple color by job (hash)
    def job_color(job:str):
        import random
        random.seed(hash(job) % (2**32))
        return (random.random()*0.6+0.2, random.random()*0.6+0.2, random.random()*0.6+0.2)

    for _, r in df.iterrows():
        y = m2row[r["machine"]]
        ax.broken_barh([(r["start"], r["finish"]-r["start"])], (y-0.4, 0.8),
                       facecolor=job_color(r["job"]), edgecolor="black", linewidth=0.5)
        # optional label: job-op
        ax.text((r["start"]+r["finish"])/2, y, f'J{r["job"]}-O{int(r["op"])}',
                va="center", ha="center", fontsize=8, color="black")

    ax.set_yticks(list(m2row.values()))
    ax.set_yticklabels(list(m2row.keys()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(f"Gantt — {case}")
    ax.set_ylim(-0.8, len(machines)-0.2)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def plot_util(df: pd.DataFrame, case: str, path: Path):
    makespan = float(df["finish"].max())
    util = (df.groupby("machine")["duration"].sum()/makespan).reset_index(name="util")
    util = util.sort_values("machine")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(util["machine"], util["util"], label="Busy")
    ax.bar(util["machine"], 1-util["util"], bottom=util["util"], label="Idle")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Machine"); ax.set_ylabel("Share")
    ax.set_title(f"Utilization — {case} (Cmax={makespan:.1f})")
    for x, y in zip(util["machine"], util["util"]):
        ax.text(x, y+0.02, f"{y*100:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def plot_job_completion(df: pd.DataFrame, case: str, path: Path):
    comp = df.groupby("job")["finish"].max().reset_index(name="completion")
    comp = comp.sort_values("completion")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(comp["job"], comp["completion"])
    ax.set_xlabel("Job"); ax.set_ylabel("Completion time")
    ax.set_title(f"Job Completion — {case}")
    for x, y in zip(comp["job"], comp["completion"]):
        ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def plot_wip(df: pd.DataFrame, case: str, path: Path):
    events = []
    for _, r in df.iterrows():
        events.append((float(r["start"]), +1))
        events.append((float(r["finish"]), -1))
    events.sort()
    times, wip, cur = [], [], 0
    for t, d in events:
        times.append(t); wip.append(cur)
        cur += d
        times.append(t); wip.append(cur)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.step(times, wip, where="post")
    ax.set_xlabel("Time"); ax.set_ylabel("WIP")
    ax.set_title(f"WIP — {case}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

# ---------- CLI ----------
def main():
    idx = input("请输入案例编号 (例如 1 表示 mk01): ").strip() or "1"
    case = f"mk{int(idx):02d}"

    df = load_schedule(case)
    out_dir = Path("outputs") / case / "figs"
    # Draw & Save
    plot_gantt(df, case, out_dir / f"gantt_{case}.png")
    plot_util(df, case, out_dir / f"util_{case}.png")
    plot_job_completion(df, case, out_dir / f"job_completion_{case}.png")
    plot_wip(df, case, out_dir / f"wip_{case}.png")
    print(f"[OK] Saved figures to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()