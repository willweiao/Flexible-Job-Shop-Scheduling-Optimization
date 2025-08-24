"""
本项目是fjsp_makespan.py —— 灵活作业车间调度（FJSP）建模与求解
使用 Pyomo 构建 FJSP 的 MILP 数学模型，以最小化最大完工时间（Cmax）为目标。
模型包括机器分配变量、相邻序列变量以及 MTZ 防子环约束，并支持调用 Gurobi 求解器。
运行后可导出调度结果，用于后续可视化分析（如甘特图、机器利用率等）。
"""

# import dependencies
import json
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary,
    Objective, Constraint, minimize, SolverFactory, value
)

# ===== 读取 routing.csv 并构建基础数据 =====
def load_routing(routing_csv: Path):
    """
    读取 routing.csv，返回：
    - ops: [(i, j, k)]  操作索引 i 及其 (job, op)
    - eligible: {i: [m,...]}  每个 i 的可选机器
    - proc_time: {(i,m): p}   工时
    - pred: {i: i_prev or None}
    - machines: sorted list of all machines
    - jk2i: {(j,k): i}
    """
    df = pd.read_csv(routing_csv)
    jk2i = {}
    for _, r in df.iterrows():
        j, k = int(r["job_id"]), int(r["op_id"])
        jk2i[(j, k)] = len(jk2i) + 1  # 连续 1..N 的操作索引

    ops, eligible, proc_time, pred = [], {}, {}, {}
    for _, r in df.iterrows():
        j, k = int(r["job_id"]), int(r["op_id"])
        i = jk2i[(j, k)]
        ops.append((i, j, k))

        em = json.loads(r["eligible_machines"])
        pt = json.loads(r["proc_time_json"])  # {str(m): p}
        eligible[i] = sorted(int(m) for m in em)
        for ms, p in pt.items():
            proc_time[(i, int(ms))] = float(p)

        pred[i] = None if k == 1 else jk2i[(j, k - 1)]

    machines = sorted({m for em in eligible.values() for m in em})
    return ops, eligible, proc_time, pred, machines, jk2i

# 在 MILP 的 LP 松弛里，二元 x 变成分数，LP 实际上是在解一个“分数指派 + 机器容量”的线性松弛。
# 在完全可分配（允许拆分）且无排序冲突的理想情况下，机器容量逼出的最小完工时间
def compute_assignment_LB(ops, eligible, proc_time, machines):
    O = [iop for (iop,_,_) in ops]
    model = ConcreteModel()
    model.O = Set(initialize=O)
    model.M = Set(initialize=machines)
    # 连续松弛变量（允许分数指派）
    X_index = [(op,i) for op in O for i in eligible[op]]
    model.x = Var(X_index, domain=NonNegativeReals)
    model.T = Var(domain=NonNegativeReals)

    def assign_rule(m, op):
        return sum(m.x[(op,i)] for i in eligible[op]) == 1
    model.Assign = Constraint(model.O, rule=assign_rule)

    def cap_rule(m, i):
        return sum(proc_time[(op,i)]*m.x[(op,i)] for op in O if (op,i) in m.x) <= m.T
    model.Cap = Constraint(model.M, rule=cap_rule)

    model.OBJ = Objective(expr=model.T, sense=minimize)
    SolverFactory("gurobi").solve(model, tee=False)
    return float(value(model.T))

# ===== 计算紧化用下界 ES 和总上界 H =====
def bounds_for_bigM(ops, eligible, proc_time):
    """
    返回：
    - pmin: {i: min_m p(i,m)}
    - ES:   {i: job内最早开工下界（按最短机时前缀）}
    - H:    makespan 的粗上界（取 sum pmin * 1.4）
    """
    pmin = {i: min(proc_time[(i, m)] for m in eligible[i]) for (i, _, _) in ops}

    ES, pref = {}, defaultdict(float)
    for _, j, k in sorted(ops, key=lambda t: (t[1], t[2])):
        # 找到 i（按 (j,k) 回推）
        i = next(ii for (ii, jj, kk) in ops if jj == j and kk == k)
        if k == 1:
            pref[(j, k)] = 0.0
        else:
            i_prev = next(ii for (ii, jj, kk) in ops if jj == j and kk == k - 1)
            pref[(j, k)] = pref[(j, k - 1)] + pmin[i_prev]
        ES[i] = pref[(j, k)]

    H = sum(pmin.values()) * 1.4  # 放松系数可调 1.3~1.6
    return pmin, ES, H


# ===== 生成“相邻弧”结构 =====
def build_adj_structures(I, M, eligible, ES, H):
    """
    相邻序列建模所需的结构：
    - Vi: {i -> [op,...]}           能在机器 i 上加工的操作集合
    - arcs_all: [(i,a,b), ...]      机器 i 上的相邻弧（含 SRC/SNK）
    - arcs_opop: [(i,a,b), ...]     仅操作->操作的相邻弧（用于时间 Big-M 与 MTZ）
    - in_keys / incoming:           入度约束索引与映射 ((i,b) -> list of a)
    - out_keys / outgoing:          出度约束索引与映射 ((i,a) -> list of b)
    - Mbig: {(i,a,b): ...}          每条操作->操作弧的 Big-M
    - Ui: {i: |Vi[i]|}              MTZ 的上界
    """
    # 机器 i 上可加工的操作集合 Vi
    Vi = {i: sorted([op for op in I if i in eligible[op]]) for i in M}

    arcs_all, arcs_opop = [], []
    incoming, in_keys = {}, []
    outgoing, out_keys = {}, []

    for i in M:
        nodes_src = ['SRC'] + Vi[i]      # δ_i 用 'SRC' 表示
        nodes_dst = Vi[i] + ['SNK']      # ε_i 用 'SNK' 表示

        # 所有相邻弧（含 SRC/SNK）
        for a in nodes_src:
            for b in nodes_dst:
                if a == b:
                    continue
                arcs_all.append((i, a, b))
                if isinstance(a, int) and isinstance(b, int):
                    arcs_opop.append((i, a, b))

        # 入弧列表：对每个 b（真实操作）统计来自 SRC 或其它操作的弧
        for b in Vi[i]:
            in_keys.append((i, b))
            incoming[(i, b)] = [a for a in (['SRC'] + Vi[i]) if a != b]

        # 出弧列表：对每个 a（真实操作）统计指向其它操作或 SNK 的弧
        for a in Vi[i]:
            out_keys.append((i, a))
            outgoing[(i, a)] = [b for b in (Vi[i] + ['SNK']) if b != a]

    # Big-M：只对 操作->操作 弧定义；紧化用 H - ES[b]
    Mbig = {(i, a, b): max(1.0, H - ES[b]) for (i, a, b) in arcs_opop}

    # MTZ 上界：每台机上最多 |Vi[i]| 个节点
    Ui = {i: max(1, len(Vi[i])) for i in M}

    return Vi, arcs_all, arcs_opop, in_keys, incoming, out_keys, outgoing, Mbig, Ui

# ===== 建 Pyomo 模型 =====
def build_model_adj(ops, eligible, proc_time, pred, machines, ES, H,
                    Vi, arcs_all, arcs_opop, in_keys, incoming, out_keys, outgoing, Mbig, Ui):
    """
    相邻弧 + MTZ 防子环 的 Pyomo 模型：
      变量：x, S, C, Cmax, y(相邻弧), u(MTZ序)
      约束：机器唯一、完成时间、工艺先后、相邻弧时间、入度=指派、出度=指派、MTZ 防子环、makespan
    """
    model = ConcreteModel()

    # === 基本集合/参数 ===
    I = sorted([i for (i, _, _) in ops])    # 操作集合（整数索引）
    M = machines                             # 机器集合
    model.I = Set(initialize=I)
    model.M = Set(initialize=M)

    # 每个操作的可行机集合（Python dict 持有）
    Mi = {i: set(eligible[i]) for i in I}

    # 工时字典（Python 侧持有）
    p_im = {(i, m): proc_time[(i, m)] for i in I for m in Mi[i]}

    # 相邻弧集合
    model.ArcsAll  = Set(dimen=3, initialize=arcs_all)   # (i,a,b), a/b 可为 'SRC'/'SNK' 或 op
    model.ArcsOpOp = Set(dimen=3, initialize=arcs_opop)  # 仅 op->op 的弧

    # 入/出度约束索引（二维）
    model.InKeys   = Set(dimen=2, initialize=in_keys)    # (i,b)
    model.OutKeys  = Set(dimen=2, initialize=out_keys)   # (i,a)

    # Big-M 参数（只对 op->op 弧）
    model.Mbig = Param(model.ArcsOpOp, initialize=Mbig, within=NonNegativeReals, mutable=False)

    # === 变量 ===
    # 机器指派
    model.x = Var(((op, i) for op in I for i in Mi[op]), domain=Binary)
    # 开工/完工/最大完工
    model.S = Var(model.I, domain=NonNegativeReals)
    model.C = Var(model.I, domain=NonNegativeReals)
    model.Cmax = Var(domain=NonNegativeReals)
    # 相邻弧选择
    model.y = Var(model.ArcsAll, domain=Binary)
    # MTZ 序变量（机级）：索引为 (i, op) 仅对真实操作
    UIndex = [(i, op) for i in M for op in Vi[i]]
    model.UIndex = Set(dimen=2, initialize=UIndex)
    # 机器是否被使用的二元变量（确保每台机至多/恰好一条链）
    model.w = Var(model.M, domain=Binary)

    def u_bounds(m, i, op):
        return (0.0, float(Ui[i]))
    model.u = Var(model.UIndex, domain=NonNegativeReals, bounds=u_bounds)

    # === 约束 ===

    # 1) 机器唯一指派
    def one_machine_rule(m, op):
        return sum(m.x[(op, i)] for i in Mi[op]) == 1
    model.OneMachine = Constraint(model.I, rule=one_machine_rule)

    # 2) 完成时间（等式写法）
    def completion_rule(m, op):
        return m.C[op] == m.S[op] + sum(p_im[(op, i)] * m.x[(op, i)] for i in Mi[op])
    model.Completion = Constraint(model.I, rule=completion_rule)

    # 3) 工艺先后（同作业）
    def precedence_rule(m, op):
        if pred[op] is None:
            return Constraint.Skip
        return m.S[op] >= m.C[pred[op]]
    model.Precedence = Constraint(model.I, rule=precedence_rule)

    # 4) 相邻弧的时间约束（仅对 操作->操作 弧）
    def adj_time_rule(m, i, a, b):
        # S_b >= S_a + p_{i,a} - M * (1 - y_{i,a,b})
        return m.S[b] >= m.S[a] + p_im[(a, i)] - m.Mbig[(i, a, b)] * (1 - m.y[(i, a, b)])
    model.AdjTime = Constraint(model.ArcsOpOp, rule=adj_time_rule)

    # 5) 入度 = 指派：每个被分配到 i 的操作 b，恰有 1 条入弧（来自 SRC 或其它操作）
    #    sum_{a in incoming[(i,b)]} y_{i,a,b} = x_{b,i}
    model._incoming = incoming  # python 挂载，供规则访问
    def indeg_rule(m, i, b):
        return sum(m.y[(i, a, b)] for a in m._incoming[(i, b)]) == m.x[(b, i)]
    model.InDeg = Constraint(model.InKeys, rule=indeg_rule)

    # 6) 出度 = 指派：每个被分配到 i 的操作 a，恰有 1 条出弧（指向其它操作或 SNK）
    #    sum_{b in outgoing[(i,a)]} y_{i,a,b} = x_{a,i}
    model._outgoing = outgoing
    def outdeg_rule(m, i, a):
        return sum(m.y[(i, a, b)] for b in m._outgoing[(i, a)]) == m.x[(a, i)]
    model.OutDeg = Constraint(model.OutKeys, rule=outdeg_rule)

    # 7) MTZ 防子环（base on machine）
    #    u_{i,b} >= u_{i,a} + 1 - Ui[i]*(1 - y_{i,a,b})，仅对 操作->操作 弧
    def mtz_rule(m, i, a, b):
        return m.u[(i, b)] >= m.u[(i, a)] + 1 - Ui[i] * (1 - m.y[(i, a, b)])
    model.MTZ = Constraint(model.ArcsOpOp, rule=mtz_rule)
    
    # 8) makespan：对每个作业末工序
    jobs = {}
    for iop, j, k in ops:
        jobs.setdefault(j, []).append((k, iop))
    i_end_list = [iop for j, ks in jobs.items() for (_, iop) in [max(ks)]]

    # 以下9）~ 12）约束是新增的为了防止出现重叠并行任务
    # 9）源点 SRC 的“起点数” = w[i]（有链才有一个起点）
    def src_degree_rule(m, i):
        terms = [m.y[(i, 'SRC', b)]
                 for (ii, a, b) in m.ArcsAll
                 if ii == i and a == 'SRC' and b != 'SNK']
        if len(terms) == 0:
            # 这台机本身就没有可加工的操作
            return m.w[i] == 0
        return sum(terms) == m.w[i]
    model.SrcDegree = Constraint(model.M, rule=src_degree_rule)

    # 10）汇点 SNK 的“终点数” = w[i]
    def snk_degree_rule(m, i):
        terms = [m.y[(i, a, 'SNK')]
                 for (ii, a, b) in m.ArcsAll
                 if ii == i and b == 'SNK' and a != 'SRC']
        if len(terms) == 0:
            return m.w[i] == 0
        return sum(terms) == m.w[i]
    model.SnkDegree = Constraint(model.M, rule=snk_degree_rule)

    # 11）x ⇒ w（有任意指派则 w[i] 必须为 1）
    def x_implies_w_rule(m, i):
        Vi_size = len(Vi[i])
        if Vi_size == 0:
            return m.w[i] == 0
        return sum(m.x[(op, i)] for op in Vi[i]) <= Vi_size * m.w[i]
    model.XImpliesW = Constraint(model.M, rule=x_implies_w_rule)

    # 12）w ⇒ x（w[i]=1 时至少有一个指派，避免“空链”）
    def w_implies_x_rule(m, i):
        if len(Vi[i]) == 0:
            return m.w[i] == 0
        return m.w[i] <= sum(m.x[(op, i)] for op in Vi[i])
    model.WImpliesX = Constraint(model.M, rule=w_implies_x_rule)

    # 以下两个约束13）和14）是为了加强模型下界，减少剪枝的复杂度
    # 13）机器负荷下界：每台机 i：Σ_{op∈Vi[i]} p_{i,op} * x[(op,i)] <= Cmax
    def load_rule(m, i):
        # Vi 和 p_im 是 build_model_adj 里已有的局部变量（上文构造过）
        return sum(p_im[(op, i)] * m.x[(op, i)] for op in Vi[i]) <= m.Cmax
    model.MachineLoad = Constraint(model.M, rule=load_rule)

    # 14）最早开始/最早完成的显式下界：把预处理得到的 ES（earliest start）加进模型，能帮 LP 找到更紧的时间窗口
    def es_rule(m, op):
        return m.S[op] >= ES[op]   # ES 作为 build_model_adj 的入参已经传进来
    model.EarliestStart = Constraint(model.I, rule=es_rule)

    jobs = {}
    for iop, j, k in ops:
        jobs.setdefault(j, []).append((k, iop))
    i_end_list = [iop for j, ks in jobs.items() for (_, iop) in [max(ks)]]

    def makespan_rule(m, i_end):
        return m.Cmax >= m.C[i_end]
    model.Makespan = Constraint(i_end_list, rule=makespan_rule)

    # 目标
    model.OBJ = Objective(expr=model.Cmax, sense=minimize)

    # 便于外部使用的挂件
    model._Mi = Mi
    model._p_im = p_im
    model._Vi = Vi

    return model


# ===== Warm Start：给初值和相邻弧 =====
def warm_start(model, ops, eligible, proc_time):
    # x：每个操作分配到最短加工时间的机器
    for (i, _, _) in ops:
        m_star = min(eligible[i], key=lambda m: proc_time[(i, m)])
        for m in eligible[i]:
            if (i, m) in model.x:
                model.x[(i, m)].value = 1.0 if m == m_star else 0.0

    #  S/C：按作业顺序前推（不考虑同机冲突，仅给一个粗初值）
    cur = defaultdict(float)
    for _, j, k in sorted(ops, key=lambda t: (t[1], t[2])):
        i = next(ii for (ii, jj, kk) in ops if jj == j and kk == k)
        m_star = min(eligible[i], key=lambda m: proc_time[(i, m)])
        p = proc_time[(i, m_star)]
        model.S[i].value = cur[j]
        model.C[i].value = cur[j] + p
        cur[j] += p

    model.Cmax.value = max(value(model.C[i]) for (i, _, _) in ops)


def warm_start_adj(model, ops, eligible, proc_time):
    from collections import defaultdict
    by_m = defaultdict(list)
    for (iop,_,_) in ops:
        ms = [m for m in eligible[iop] if (iop,m) in model.x and (model.x[(iop,m)].value or 0)>0.5]
        m_star = ms[0] if ms else min(eligible[iop], key=lambda mm: proc_time[(iop,mm)])
        s_val = model.S[iop].value if model.S[iop].value is not None else 0.0
        by_m[m_star].append((s_val, iop))
    for m, lst in by_m.items():
        lst.sort()
        if lst and (m,'SRC',lst[0][1]) in model.y: model.y[(m,'SRC',lst[0][1])].value = 1.0
        for (_,a),(_,b) in zip(lst, lst[1:]):
            if (m,a,b) in model.y: model.y[(m,a,b)].value = 1.0
        if lst and (m,lst[-1][1],'SNK') in model.y: model.y[(m,lst[-1][1],'SNK')].value = 1.0
    # u 初值可按序号给：u[(m,op)]=rank
    for m,lst in by_m.items():
        for rank,(_,op) in enumerate(sorted(lst)):
            if (m,op) in model.u: model.u[(m,op)].value = float(rank)


# =====取解与快速可行性检查 =====
def extract_schedule(model, ops, eligible):
    # 提取每个操作的分配机台与时间
    assignment = {}
    for (i, _, _) in ops:
        ms = [m for m in eligible[i] if (i, m) in model.x and value(model.x[(i, m)]) > 0.5]
        assignment[i] = ms[0] if ms else None

    schedule = {i: (value(model.S[i]), value(model.C[i]), assignment[i]) for (i, _, _) in ops}

    # 简单同机不重叠验算
    by_m = defaultdict(list)
    for (i, _, _) in ops:
        s, c, m = schedule[i]
        by_m[m].append((s, c, i))
    for m, lst in by_m.items():
        lst.sort()
        for (s1, c1, i1), (s2, c2, i2) in zip(lst, lst[1:]):
            assert s2 + 1e-6 >= c1, f"Overlap on machine {m}: {i1}->{i2}"

    return schedule


# ===== 8) 保存结果进行数据分析 =====
def export_gantt(model, ops, eligible, out_dir="outputs/mk01"):
    """
    排程甘特图数据（按机器的一行一条区间）
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # 取分配的机器
    assign = {}
    for (i, _, _) in ops:
        ms = [m for m in eligible[i] if (i, m) in model.x and value(model.x[(i,m)]) > 0.5]
        assign[i] = ms[0] if ms else None

    rows = []
    for (i, j, k) in ops:
        m = assign[i]
        s = float(value(model.S[i]))
        c = float(value(model.C[i]))
        rows.append({
            "machine": m, "job": j, "op": k,
            "start": s, "finish": c, "duration": c - s,
            "op_id": i
        })
    df = pd.DataFrame(rows).sort_values(["machine", "start", "finish"])
    df.to_csv(out / "schedule_gantt.csv", index=False, encoding="utf-8")
    print("Saved:", (out / "schedule_gantt.csv").resolve())

def export_assignment_long(model, ops, eligible, out_dir="outputs/mk01"):
    """
    长表（便于直接透视/分组）
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (i, j, k) in ops:
        for m in eligible[i]:
            val = float(value(model.x[(i,m)])) if (i,m) in model.x else 0.0
            rows.append({"op_id": i, "job": j, "op": k, "machine": m, "x": val})
    pd.DataFrame(rows).to_csv(out/"assignment_long.csv", index=False)

def export_assignment_wide(model, ops, eligible, out_dir="outputs/mk01"):
    """
    宽表（每个操作一行，附带所选机器）
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (i,j,k) in ops:
        chosen = [m for m in eligible[i] if (i,m) in model.x and value(model.x[(i,m)])>0.5]
        rows.append({"op_id": i, "job": j, "op": k, "machine": chosen[0] if chosen else None,
                     "start": float(value(model.S[i])), "finish": float(value(model.C[i]))})
    pd.DataFrame(rows).to_csv(out/"assignment_wide.csv", index=False)

def export_metadata(model, result, out_dir="outputs/mk01", extra=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {
        "Cmax": float(value(model.Cmax)),
        "termination": str(result.solver.termination_condition),
        "time": getattr(result.solver, "time", None),
        "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S"),
        "solver": "gurobi",
        "params": {
            # 若你设置了参数，也可以一起存下来
            "TimeLimit": 300, "MIPGap": 0.02, "Threads":16,
            "MIPfocus": 1, "Heuristic": 0.2, "Presolve": 2,
            "Cuts": 2
        }
    }
    if extra: meta.update(extra)
    (out/"run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", (out/"run_meta.json").resolve())


# ===== 9) 主程序：把以上步骤串起来 =====
def main():
    case_name = "mk02"  # 根据所运行的案例进行修改
    OUT_DIR = Path(f"outputs/{case_name}")
    LOGS_DIR = Path("logs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 读取数据
    routing_csv = Path(f"data/processed/brandimarte/{case_name}/routing.csv")
    assert routing_csv.exists(), f"routing.csv not found: {routing_csv}"

    # 2）建模
    ops, eligible, proc_time, pred, machines, jk2i = load_routing(routing_csv)
    pmin, ES, H = bounds_for_bigM(ops, eligible, proc_time)
    I_ops = [i for (i,_,_) in ops]
    Vi, arcs_all, arcs_opop, in_keys, incoming, out_keys, outgoing, Mbig, Ui = \
        build_adj_structures(I=I_ops, M=machines, eligible=eligible, ES=ES, H=H)
    
    # 为了更有效地推下界，考虑基于 routing.csv 的数据计算每个作业的最短加工时间之和并取最大，作为全局 Cmax 的下界
    # ops: list of (op_id, job, ord), eligible: {op_id: [machines]}, proc_time: {(op_id, m): p}
    # 按作业分组
    job_ops = {}
    for iop, j, k in ops:
        job_ops.setdefault(j, []).append(iop)

    # 每个操作的“最短可加工时间”（在可行机里取 min）
    op_min_p = {}
    for j, oplist in job_ops.items():
        for op in oplist:
            op_min_p[op] = min(proc_time[(op, m)] for m in eligible[op])

    # 每个作业的最短总工时 & 全局下界
    LB_job = {j: sum(op_min_p[op] for op in oplist) for j, oplist in job_ops.items()}
    Cmax_lb_jobs = max(LB_job.values())

    # 计算机器容量分配的赋值松弛下界（preemptive的情形）
    LB_assign = compute_assignment_LB(ops, eligible, proc_time, machines)

    model = build_model_adj(ops, eligible, proc_time, pred, machines, ES, H,
                        Vi, arcs_all, arcs_opop, in_keys, incoming, out_keys, outgoing, Mbig, Ui)
    model.Cmax.setlb(max(Cmax_lb_jobs, int(LB_assign + 0.999)))

    # 3) Warm Start
    warm_start(model, ops, eligible, proc_time)
    
    # 4) 导出 LP / MPS 到指定目录
    model.write(str(OUT_DIR / "model.mps"), format="mps")

    # 5）调用Gurobi求解
    solver = SolverFactory("gurobi")
    solver.options["LogFile"]       = str(LOGS_DIR / f"{case_name}_gurobi.log")
    solver.options["TimeLimit"]     = 300
    solver.options["MIPGap"]        = 0.02
    solver.options["Threads"]       = 16
    solver.options["MIPFocus"]      = 1
    solver.options["Heuristics"]    = 0.2
    solver.options["Presolve"]      = 2
    solver.options["Cuts"]          = 2
    solver.options["NodefileStart"] = 0.5                     # 因为求解过程过长因此考虑中途保存记录

    solver.options["ResultFile"] = str(OUT_DIR / "solution.sol")

    result = solver.solve(model, tee=True)
    print("Cmax =", value(model.Cmax))
    print("LP/MPS/日志/解文件已导出到：", OUT_DIR.resolve(), LOGS_DIR.resolve())

    # 5) 输出甘特 / 赋值表 / 元数据
    export_gantt(model, ops, eligible, out_dir=OUT_DIR)
    export_assignment_long(model, ops, eligible, out_dir=OUT_DIR)
    export_assignment_wide(model, ops, eligible, out_dir=OUT_DIR)
    export_metadata(model, result, out_dir=OUT_DIR,
                    extra={"lp": str((OUT_DIR/"model.lp").resolve()),
                           "mps": str((OUT_DIR/"model.mps").resolve()),
                           "sol": str((OUT_DIR/"solution.sol").resolve()),
                           "log": str((LOGS_DIR/f"{case_name}_gurobi.log").resolve())})

if __name__ == "__main__":
    main()