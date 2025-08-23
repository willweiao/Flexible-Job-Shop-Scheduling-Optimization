import re, json
import pandas as pd
from pathlib import Path

# === config ===
RAW_DIR = Path("data/raw/brandimarte")          # 原始 mkXX 文件所在目录
OUT_DIR = Path("data/processed/brandimarte")    # 处理后的文件保存目录
PATTERN = r"mk(\d+)\.txt"           # 匹配文件名模式 (mk01.txt, mk02.txt ...)
FORCE = False                       # 已存在是否覆盖

# === process data ===
def numeric_key(p: Path):
    """按 mk 文件编号排序"""
    m = re.search(PATTERN, p.name, flags=re.I)
    return int(m.group(1)) if m else 10**9

def read_tokens(path: Path):
    """读取文件并拆成整数 token 列表"""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    toks = re.findall(r"-?\d+", data)
    return list(map(int, toks))

def parse_mk_file(path: Path):
    """解析单个 mk 文件"""
    toks = read_tokens(path)
    it = iter(toks)

    def nxt():  # 安全取 token, 取下一个整数
        try:
            return next(it)
        except StopIteration:
            raise ValueError(f"Unexpected EOF while parsing {path.name}")

    J = nxt(); M = nxt()   # 作业数 & 机器数
    rows = []
    total_ops = 0

    for j in range(1, J+1):
        K = nxt()  # 工序数
        for k in range(1, K+1):
            q = nxt()  # 可选机器数
            machines = []
            ptime = {}
            for _ in range(q):
                m = nxt(); p = nxt()
                machines.append(m)          
                ptime[str(m)] = int(p)
            # 校验
            assert len(machines) == q and len(ptime) == q
            rows.append({
                "job_id": j,                                                    # 按顺序生成job id，从1开始
                "op_id": k,                                                     # 同样按顺序为每个operation编号，从1开始
                "eligible_machines": json.dumps(machines, ensure_ascii=False),
                "proc_time_json": json.dumps(ptime, ensure_ascii=False),
            })
            total_ops += 1

    return J, M, total_ops, rows

def process_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in RAW_DIR.iterdir() if re.search(PATTERN, p.name, re.I)],
                   key=numeric_key)

    if not files:
        print(f"No files matching {PATTERN} found in {RAW_DIR}")
        return

    for path in files:
        base = path.stem  # file mk01作为base
        out_sub = OUT_DIR / base
        if out_sub.exists() and not FORCE:
            print(f"[SKIP] {base} exists. Use FORCE=True to overwrite.")
            continue
        out_sub.mkdir(parents=True, exist_ok=True)

        try:
            J, M, total_ops, rows = parse_mk_file(path)
        except Exception as e:
            print(f"[ERROR] parsing {path.name}: {e}")
            continue

        df = pd.DataFrame(rows)
        df.to_csv(out_sub / "routing.csv", index=False, encoding="utf-8")

        avg_choices = df["eligible_machines"].apply(lambda s: len(json.loads(s))).mean()
        print(f"[OK] {base}: J={J}, M={M}, |O|={total_ops}, avg|M_i|={avg_choices:.2f}")

# === main ====
if __name__ == "__main__":
    process_all()