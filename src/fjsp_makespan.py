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
import math
import pandas as pd
import pyomo.environ as pyo
from math import ceil, isfinite
from pyomo.opt import TerminationCondition as TC, SolverStatus as SS
from pyomo.core import TransformationFactory 
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
    model.Mbig = Param(model.ArcsOpOp, initialize=Mbig, within=NonNegativeReals, mutable=True)

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


# =====Warm Start：从模型中提取 LP 松弛解=====
def solve_lp_relaxation_and_extract(model, ops, eligible, out_dir: Path, case_name: str, timelimit=30, threads=16):
    """
    生成并求解 LP 放松；返回 (ok, relaxed, resLP, Sstar, xstar, obj_lp)。
    同时把 LP 变量值落盘：Sstar_xstar_lp.csv。
    """
    # 1) 生成 LP 放松副本
    try:
        tf = TransformationFactory('core.relax_integer_vars')
    except Exception:
        tf = TransformationFactory('core.relax_int_vars')
    relaxed = tf.create_using(model)

    # 2) 求解 LP
    lp_solver = SolverFactory("gurobi")
    lp_solver.options.update({"TimeLimit": timelimit, "Presolve": 2, "Threads": threads, "LogFile": str(out_dir / f"{case_name}_lp.log")})
    resLP = lp_solver.solve(relaxed, tee=False)

    # 3) 取目标值
    obj_comp = next((obj for obj in relaxed.component_objects(Objective, active=True)), None)
    obj_lp = None if obj_comp is None else float(value(obj_comp.expr))

    # 4) 判定状态
    tc = getattr(resLP.solver, "termination_condition", None)
    ok = tc in {TC.optimal, TC.feasible, TC.maxTimeLimit}
    if not ok:
        print(f"[LP] termination={tc}, obj={obj_lp}  -> LP 可能没成功。")

    # 5) 提取 S*/x*
    Sstar = {}
    for (op,_,_) in ops:
        try:
            Sstar[op] = float(value(relaxed.S[op]) or 0.0)
        except Exception:
            Sstar[op] = 0.0

    xstar = {}
    for (op,_,_) in ops:
        for i in eligible[op]:
            try:
                xstar[(op,i)] = float(value(relaxed.x[(op,i)]) or 0.0)
            except Exception:
                xstar[(op,i)] = 0.0

    # 6) 落盘检查
    rows = []
    for (op, j, k) in ops:
        rows.append({"op_id": op, "job": j, "op": k, "S_star": Sstar.get(op, 0.0)})
        for i in eligible[op]:
            rows.append({"op_id": op, "job": j, "op": k, "machine": i, "x_star": xstar.get((op,i), 0.0)})
    df = pd.DataFrame(rows)
    (out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "Sstar_xstar_lp.csv", index=False, encoding="utf-8")
    print(f"[LP] ok={ok}, obj={obj_lp}; S*/x* 已导出: {(out_dir /'Sstar_xstar_lp.csv').resolve()}")

    return ok, relaxed, resLP, Sstar, xstar, obj_lp

# LP 引导 + SSGS：构造严格可行的初解
def lp_round_to_feasible(ops, eligible, proc_time, pred, machines,
                         Sstar=None, xstar=None, tie_eps=1e-6):
    """
    用 LP 的 (S*, x*) 引导 SSGS，构造可行解：
      - 先按 x* 确定每个操作的机器指派（平局用更短 p 打破）；
      - 再用 SSGS 按 S* 的先后把操作逐个安排，保证：前后继、同机不重叠、C=S+p。
    返回: (UB, x0, S0, C0)
    """
    # 0) LP 值兜底
    if Sstar is None:  # 若调用方没传，就用 0
        Sstar = {op:0.0 for (op,_,_) in ops}
    if xstar is None:
        xstar = {(op,i):0.0 for (op,_,_) in ops for i in eligible[op]}

    # 1) 机器指派 x0：argmax_i x*(op,i)，平局用更小加工时间 p 打破
    x0 = {}
    for (op,_,_) in ops:
        cand = []
        for i in eligible[op]:
            score = xstar.get((op,i), 0.0)
            cand.append((score, -proc_time[(op,i)], i))  # 分数大优先，p 小优先
        # 若全 0，则按最短 p 选一台
        if not cand:
            raise RuntimeError(f"op {op} has empty eligible set.")
        score_max = max(c[0] for c in cand)
        if score_max < tie_eps:
            i_hat = min(eligible[op], key=lambda i: proc_time[(op,i)])
        else:
            # 先按分数降序，再按 p 升序
            cand.sort(key=lambda t: (t[0], t[1]), reverse=False)  # 因为第二项是 -p
            i_hat = max(cand, key=lambda t: (t[0], -t[1]))[2]    # 分数最大，若平局 p 最小
        x0[op] = i_hat

    # 2) 构造后继表与入度（基于 pred）
    succ = {}
    indeg = {op: 0 for (op,_,_) in ops}
    for (op,_,_) in ops:
        p = pred.get(op, None)
        if p is not None:
            succ.setdefault(p, []).append(op)
            indeg[op] += 1

    # 3) SSGS：机器就绪时间 & 作业就绪时间
    RM = {i: 0.0 for i in machines}   # machine ready time
    RJ = {}
    job_of = {op: j for (op,j,_) in ops}
    for _,j,_ in ops:
        RJ.setdefault(j, 0.0)

    # 初始就绪池：入度为 0 的首工序，按 S* 早的排前
    Q = [op for (op,_,_) in ops if indeg[op] == 0]
    Q.sort(key=lambda o: Sstar.get(o, 0.0))

    S0, C0 = {}, {}
    while Q:
        o = Q.pop(0)
        i_hat = x0[o]
        s = max(RM[i_hat], RJ[job_of[o]])
        p = proc_time[(o, i_hat)]
        S0[o] = s
        C0[o] = s + p
        RM[i_hat] = C0[o]
        RJ[job_of[o]] = C0[o]
        for b in succ.get(o, []):
            indeg[b] -= 1
            if indeg[b] == 0:
                # 维持按 S* 的优先顺序插入
                insert_at = 0
                Sb = Sstar.get(b, 0.0)
                while insert_at < len(Q) and Sstar.get(Q[insert_at], 0.0) <= Sb:
                    insert_at += 1
                Q.insert(insert_at, b)

    UB = max(C0.values()) if C0 else 0.0
    return UB, x0, S0, C0

def _clip(val, lo=None, hi=None):
    if val is None: return None
    if lo is not None: val = max(lo, val)
    if hi is not None: val = min(hi, val)
    return val

def apply_warm_start(model, x0, S0, C0, UB=None):
    """
    把可行解灌进 Pyomo 变量，并把 S/C/Cmax 夹到模型边界内，避免 W1002。
    """
    # x
    for key in model.x:
        op, m = key
        v = 1.0 if (op in x0 and x0[op] == m) else 0.0
        model.x[key].value = v
        model.x[key].stale = False

    # 上界用于裁剪
    cmax_lb = model.Cmax.lb if model.Cmax.has_lb() else 0.0
    cmax_ub = model.Cmax.ub if model.Cmax.has_ub() else None

    # S, C
    for op in model.I:
        if op in S0:
            sv = _clip(float(S0[op]), 0.0, cmax_ub)
            model.S[op].value = sv; model.S[op].stale = False
        if op in C0:
            cv = _clip(float(C0[op]), 0.0, cmax_ub)
            model.C[op].value = cv; model.C[op].stale = False

    # Cmax：取 max(下界, 估计 UB, 当前最大完工)
    guess = cmax_lb
    if C0:
        try: guess = max(guess, max(float(C0[o]) for o in C0))
        except Exception: pass
    if UB is not None:
        try: guess = max(guess, float(UB))
        except Exception: pass
    if cmax_ub is not None:
        guess = min(guess, float(cmax_ub))
    model.Cmax.value = guess
    model.Cmax.stale = False

# 相邻弧 & MTZ 的暖启动
def warm_start_adj(model, ops, eligible, proc_time):
    """
    度安全的 y/u MIPStart：
      - 对每台机 i，取分配在 i 的操作按 S 升序得到序列 L；
      - 对序列里每个 a，必须从 model._outgoing[(i,a)] 里选 exactly one b，
        优先选“时间表中的下一个 op”（若那条弧存在）；若不存在，则：
          * 若有 (i,a,'SNK') 就选 SNK；
          * 否则在候选 b 中选 S[b] 最接近且 >= C[a] 的（找不到则任意一个），确保 sum_out = 1；
      - 对序列里每个 b，入度也会随之满足 sum_in = 1；若首个 b 的 incoming 里没有 (i,'SRC',b)，
        则从候选 incoming 里选时间上最合理的 a；最后兜底若还没有，再设 SRC。
      - u 序变量按序号赋 0,1,2,...
    """
    # 拿 S 值、当前 x 选择
    Sval = {op: float(value(model.S[op]) or 0.0) for (op,_,_) in ops}
    Cval = {op: float(value(model.C[op]) or (Sval[op]+1.0)) for (op,_,_) in ops}
    x_pick = {}
    for (op,_,_) in ops:
        chosen = [i for i in eligible[op] if (op,i) in model.x and (value(model.x[(op,i)]) or 0.0) > 0.5]
        x_pick[op] = chosen[0] if chosen else None

    # 清零所有 y
    for key in model.ArcsAll:
        try:
            model.y[key].value = 0.0
            model.y[key].stale = False
        except Exception:
            pass

    # 按机构造序列并设置出度=1
    by_m = {}
    for (op,_,_) in ops:
        i = x_pick.get(op, None)
        if i is None: continue
        by_m.setdefault(i, []).append(op)

    for i, L in by_m.items():
        L.sort(key=lambda o: (Sval[o], Cval[o]))

        # 给 u
        if hasattr(model, "u"):
            for r, o in enumerate(L):
                key = (i, o)
                if key in model.u:
                    model.u[key].value = float(r)
                    model.u[key].stale = False

        # 逐个 a 设出度
        for idx, a in enumerate(L):
            cand_out = list(model._outgoing.get((i,a), []))  # 候选 b，可能含 'SNK'
            chosen_b = None

            # 理想选择：时间表中的“下一个” b
            if idx+1 < len(L):
                b_next = L[idx+1]
                if (i,a,b_next) in model.y:
                    chosen_b = b_next

            # 若理想弧不存在，尽量选 SNK
            if chosen_b is None and (i,a,'SNK') in model.y:
                chosen_b = 'SNK'

            # 若还没有，挑个“最接近 C[a] 的 b”
            if chosen_b is None:
                # 在 cand_out 里剔除 'SNK'/'SRC'（不该出现 SRC）
                real_bs = [b for b in cand_out if b not in ('SRC','SNK')]
                if real_bs:
                    # 选 S[b] >= C[a] 的最近者，否则 S[b] 最小者
                    later = [b for b in real_bs if Sval[b] >= Cval[a] - 1e-9]
                    if later:
                        chosen_b = min(later, key=lambda b: (Sval[b]-Cval[a], Sval[b]))
                    else:
                        chosen_b = min(real_bs, key=lambda b: Sval[b])

            # 兜底： cand_out 至少会有一个元素（建模应保证），随便选一个
            if chosen_b is None and cand_out:
                chosen_b = cand_out[0]

            # 设 y=1
            if chosen_b is not None and (i,a,chosen_b) in model.y:
                model.y[(i,a,chosen_b)].value = 1.0
                model.y[(i,a,chosen_b)].stale = False
            else:
                # 仍然没法设，打印一下方便排查
                print(f"[WS-ADJ] cannot set out-arc for (i={i}, a={a}); outgoing={cand_out}")

        # 处理首元素的入度（优先 SRC→first）
        if L:
            first = L[0]
            cand_in = list(model._incoming.get((i, first), []))
            if (i,'SRC',first) in model.y:
                model.y[(i,'SRC',first)].value = 1.0
                model.y[(i,'SRC',first)].stale = False
            else:
                # 选一个 a' 使得 C[a'] 最接近 S[first]
                real_as = [a for a in cand_in if a not in ('SRC','SNK')]
                if real_as:
                    a_best = max(real_as, key=lambda a: (Cval[a] <= Sval[first], -abs(Cval[a]-Sval[first])))
                    if (i,a_best,first) in model.y:
                        model.y[(i,a_best,first)].value = 1.0
                        model.y[(i,a_best,first)].stale = False

# 起点一致性自检
def check_warm_feasibility(ops, eligible, proc_time, pred, machines, x0, S0, C0, tol=1e-6):
    ok = True
    # 机器合法 & 唯一指派
    for (op,_,_) in ops:
        m = x0.get(op, None)
        if m is None:
            print(f"[WS-CHK] op {op} has no machine assignment"); ok=False; continue
        if m not in eligible[op]:
            print(f"[WS-CHK] op {op} assigned to ineligible machine {m}"); ok=False

    # 完成等式
    for (op,_,_) in ops:
        if op in S0 and op in C0:
            m = x0.get(op, None)
            if m is None: continue
            p = proc_time.get((op,m), None)
            if p is None:
                print(f"[WS-CHK] missing proc_time({op},{m})"); ok=False; continue
            if abs(C0[op] - (S0[op] + p)) > tol:
                print(f"[WS-CHK] op {op}: C!=S+p  C={C0[op]} S={S0[op]} p={p}"); ok=False

    # 前后继
    for (op,_,_) in ops:
        pre = pred.get(op, None)
        if pre is not None and (op in S0) and (pre in C0):
            if S0[op] + tol < C0[pre]:
                print(f"[WS-CHK] precedence violated: S[{op}]={S0[op]} < C[{pre}]={C0[pre]}"); ok=False

    # 同机不重叠
    by_m = {}
    for (op,_,_) in ops:
        m = x0.get(op, None)
        if m is None: continue
        by_m.setdefault(m, []).append(op)
    for m, oplist in by_m.items():
        seg = sorted(oplist, key=lambda o: S0[o])
        for a, b in zip(seg, seg[1:]):
            if S0[b] < C0[a] - tol:
                print(f"[WS-CHK] overlap on machine {m}: "
                      f"a={a}[{S0[a]},{C0[a]}), b={b}[{S0[b]},{C0[b]})")
                ok=False
    return ok

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

# 工具模块函数
# 是否有 incumbent（可行解）
def has_incumbent(result, model=None):
    tc = getattr(result.solver, "termination_condition", None)
    st = getattr(result.solver, "status", None)

    # 用字符串名字来兼容不同 Pyomo 版本
    tcname = getattr(tc, "name", str(tc)) if tc is not None else None

    # 明确排除无解情形
    if tcname in {"infeasible", "unbounded", "infeasibleOrUnbounded"}:
        return False

    # 只要 solver 回了 solution 列表，就算有 incumbent
    if hasattr(result, "solution") and result.solution:
        try:
            if len(result.solution) > 0:
                return True
        except Exception:
            pass

    # 接受这些“有解但未证最优/被打断”的终止情形
    # 注意用字符串名而不是枚举常量，避免 AttributeError
    if st in (SS.ok, SS.aborted) and tcname in {
        "maxTime", "maxIterations", "other", "feasible", "userInterrupt"
    }:
        return True

    # 兜底：如果已经 load_vars()，Cmax 有有限值也算有解
    if model is not None:
        try:
            v = pyo.value(model.Cmax)
            if v is not None and float(v) < float("inf"):
                return True
        except Exception:
            pass

    return False

# 弧/度一致性自检，避免“结构剪没了” 
def sanity_check(Vi, arcs_all, incoming, outgoing):
    ok = True
    # 每台机要有 SRC→* 和 *→SNK
    for i, ops_i in Vi.items():
        if not any(ii == i and a == 'SRC' for (ii, a, b) in arcs_all):
            print(f"[WARN] machine {i} has no SRC→* arcs"); ok = False
        if not any(ii == i and b == 'SNK' for (ii, a, b) in arcs_all):
            print(f"[WARN] machine {i} has no *→SNK arcs"); ok = False
        for op in ops_i:
            if not incoming.get((i, op)):
                print(f"[WARN] empty incoming for (i={i}, op={op})"); ok = False
            if not outgoing.get((i, op)):
                print(f"[WARN] empty outgoing for (i={i}, op={op})"); ok = False
    return ok

# 度约束的一致性自检，求解前跑一下它，马上能看到哪台机、哪个操作的入/出度不匹配
def check_degree_balance(model, tol=1e-6, max_print=10):
    """
    检查入度/出度是否与 x 一致：
      对所有 (i,a) & (i,b): sum_out y = x[a,i], sum_in y = x[b,i]
    仅用于 warm start 之后、solve 之前做体检。
    """
    # 索引集合（提高判断速度）
    x_idx = model.x.index_set()
    y_idx = model.y.index_set() if hasattr(model, "y") else set()

    def x_val(a, i):
        # 若 (a,i) 不是 x 的索引，返回 0
        if (a, i) in x_idx:
            v = value(model.x[a, i])
            return 0.0 if v is None else float(v)
        return 0.0

    def y_val(i, a, b):
        if hasattr(model, "y") and (i, a, b) in y_idx:
            v = value(model.y[i, a, b])
            return 0.0 if v is None else float(v)
        return 0.0

    bad = 0

    # 出度 = 指派
    for (i, a) in model.OutKeys:
        x_ai = x_val(a, i)
        sum_y = 0.0
        for b in model._outgoing.get((i, a), []):
            sum_y += y_val(i, a, b)
        if abs(sum_y - x_ai) > tol:
            print(f"[DEG-CHK][OUT] (i={i}, a={a}): sum_y={sum_y} != x={x_ai}")
            bad += 1
            if bad >= max_print:
                break

    # 入度 = 指派
    if bad < max_print:
        for (i, b) in model.InKeys:
            x_bi = x_val(b, i)
            sum_y = 0.0
            for a in model._incoming.get((i, b), []):
                sum_y += y_val(i, a, b)
            if abs(sum_y - x_bi) > tol:
                print(f"[DEG-CHK][IN ] (i={i}, b={b}): sum_y={sum_y} != x={x_bi}")
                bad += 1
                if bad >= max_print:
                    break

    if bad == 0:
        print("[DEG-CHK] degree balance OK for warm start.")
    else:
        print(f"[DEG-CHK] found {bad} degree mismatches (showing up to {max_print}).")

# 检查排程可行性以及相邻弧度约束与指派一致性（和模型里的 y 集合强相关）
def count_schedule_conflicts(ops, eligible, proc_time, pred, x0, S0, C0, tol=1e-6):
    """
    返回一个 dict：三类冲突的数量与列表：
      - bad_eq: C!=S+p 的操作列表
      - bad_pre: 违反工序先后的 (pre, op) 列表
      - bad_ovl: 同机重叠对 [(i, a, b, [Sa,Ca), [Sb,Cb))]
    """
    job_of = {op: j for (op,j,_) in ops}
    # 完成等式
    bad_eq = []
    for (op,_,_) in ops:
        m = x0.get(op, None)
        if m is None or op not in S0 or op not in C0: 
            bad_eq.append(op); continue
        p = proc_time.get((op, m), None)
        if p is None or abs(C0[op] - (S0[op] + p)) > tol:
            bad_eq.append(op)

    # 先后关系
    bad_pre = []
    for (op,_,_) in ops:
        pre = pred.get(op, None)
        if pre is None: continue
        if (op in S0) and (pre in C0):
            if S0[op] + tol < C0[pre]:
                bad_pre.append((pre, op))

    # 同机重叠
    by_m = {}
    for (op,_,_) in ops:
        i = x0.get(op, None)
        if i is not None and op in S0 and op in C0:
            by_m.setdefault(i, []).append(op)

    bad_ovl = []
    for i, oplist in by_m.items():
        seg = sorted(oplist, key=lambda o: (S0[o], C0[o]))
        for a, b in zip(seg, seg[1:]):
            if S0[b] < C0[a] - tol:
                bad_ovl.append((i, a, b, (S0[a], C0[a]), (S0[b], C0[b])))

    return {
        "bad_eq_count": len(bad_eq), "bad_eq_ops": bad_eq,
        "bad_pre_count": len(bad_pre), "bad_pre_pairs": bad_pre,
        "bad_ovl_count": len(bad_ovl), "bad_ovl_pairs": bad_ovl,
    }

def summarize_schedule_ok(ops, eligible, proc_time, pred, x0, S0, C0):
    info = count_schedule_conflicts(ops, eligible, proc_time, pred, x0, S0, C0)
    ok = (info["bad_eq_count"]==0 and info["bad_pre_count"]==0 and info["bad_ovl_count"]==0)
    if ok:
        print("[SSGS] 可行性 OK：无完成等式/先后/重叠冲突。")
    else:
        print(f"[SSGS] 不可行：C!=S+p: {info['bad_eq_count']}，先后冲突: {info['bad_pre_count']}，重叠: {info['bad_ovl_count']}")
        # 打印前几个例子
        if info["bad_eq_count"]>0:
            print("  例：", info["bad_eq_ops"][:5])
        if info["bad_pre_count"]>0:
            print("  例：", info["bad_pre_pairs"][:5])
        if info["bad_ovl_count"]>0:
            print("  例：", info["bad_ovl_pairs"][:3])
    return ok, info


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

def export_metadata(model, result, out_dir="outputs/mk01", solver_name="gurobi", solver_options=None, extra=None):
    """
    把最关键的最优性信息也写进去：best_bound / gap / status / termination
    solver_options 可把你实际传给求解器的 options 字典一并记录
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    Cmax = float(value(model.Cmax)) if value(model.Cmax) is not None else None

    # Pyomo 的 result.solver 里通常可以取到 best_bound / termination 等
    best_bd = getattr(result.solver, "best_bound", None)
    status = str(getattr(result.solver, "status", ""))           # SolverStatus
    termination = str(getattr(result.solver, "termination_condition", ""))  # TerminationCondition
    runtime = getattr(result.solver, "time", None)  # 有时是 wallclock_time / user_time，依接口而定

    gap = None
    if (Cmax is not None) and (best_bd is not None):
        try:
            gap = float((Cmax - float(best_bd)) / max(1.0, abs(Cmax)))
        except Exception:
            gap = None

    meta = {
        "Cmax": Cmax,
        "best_bound": float(best_bd) if best_bd is not None else None,
        "gap": gap,
        "status": status,
        "termination": termination,
        "time": runtime,
        "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S"),
        "solver": solver_name,
        "params": solver_options or {},  # <<- 记录你真实传给求解器的 options
    }
    if extra:
        meta.update(extra)

    (out/"run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", (out/"run_meta.json").resolve())


# ===== 9) 主程序：把以上步骤串起来 =====
def main():
    case_name = "mk05"  # 根据所运行的案例进行修改
    OUT_DIR = Path(f"outputs/{case_name}")
    LOGS_DIR = Path("logs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    warm_dir = OUT_DIR / "warmstart_lp"
    warm_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取数据
    routing_csv = Path(f"data/processed/brandimarte/{case_name}/routing.csv")
    assert routing_csv.exists(), f"routing.csv not found: {routing_csv}"
    ops, eligible, proc_time, pred, machines, jk2i = load_routing(routing_csv)

    # 2) 基础界：ES / 下界 / 安全上界
    pmin, ES, H = bounds_for_bigM(ops, eligible, proc_time)
    
    # 下界：作业最短链 + assignment 下界
    # 为了更有效地推下界，考虑基于 routing.csv 的数据计算每个作业的最短加工时间之和并取最大，作为全局 Cmax 的下界
    # ops: list of (op_id, job, ord), eligible: {op_id: [machines]}, proc_time: {(op_id, m): p}
    # 按作业分组
    job_ops = {}
    for iop, j, k in ops:
        job_ops.setdefault(j, []).append(iop)
    op_min_p = {op: min(proc_time[(op, m)] for m in eligible[op]) for oplist in job_ops.values() for op in oplist}
    LB_job = max(sum(op_min_p[op] for op in oplist) for oplist in job_ops.values())
    LB_assign = compute_assignment_LB(ops, eligible, proc_time, machines)
    Cmax_lb_init = max(LB_job, int(ceil(LB_assign)))

    # 安全上界（绝不会剪掉真解）
    UB_safe = int(ceil(sum(min(proc_time[(op, m)] for m in eligible[op]) for (op, _, _) in ops)))

    # 3) 构建相邻弧结构 + 模型（用 UB_safe 来算 Big-M）
    I_ops = [i for (i,_,_) in ops]
    Vi, arcs_all, arcs_opop, in_keys, incoming, out_keys, outgoing, Mbig, Ui = \
        build_adj_structures(I=I_ops, M=machines, eligible=eligible, ES=ES, H=H)
    
    # 结构一致性自检（非常重要）
    assert sanity_check(Vi, arcs_all, incoming, outgoing), "Arc/degree structure inconsistent."

    model = build_model_adj(
        ops, eligible, proc_time, pred, machines, ES, UB_safe,
        Vi, arcs_all, arcs_opop, in_keys, incoming, out_keys, outgoing, Mbig, Ui
    )

    # 设上下界（紧化但安全）
    model.Cmax.setlb(Cmax_lb_init)
    model.Cmax.setub(UB_safe)
    for op in model.I:
        model.S[op].setub(UB_safe)
        model.C[op].setub(UB_safe)

    # 4) 先解一个 LP 松弛，提取 S*/x*，再用 SSGS 构造严格可行起点
    okLP, relaxed, resLP, Sstar, xstar, obj_lp = solve_lp_relaxation_and_extract(model, ops, eligible, OUT_DIR, case_name, timelimit=30, threads=16)
    if not okLP:
        print("[LP] 松弛求解未成功（见 logs 以及 Sstar_xstar_lp.csv），后续暖启动可能无效。")
    else:
        print("LP求解OK")

    # 用 LP 引导的 SSGS 生成可行解；若 LP 解无效则会退化为基于 eligible/p 的 SSGS
    UB_ws, x0, S0, C0 = lp_round_to_feasible(
        ops, eligible, proc_time, pred, machines,
        Sstar=Sstar, xstar=xstar
    )
    ok_sched, sched_info = summarize_schedule_ok(ops, eligible, proc_time, pred, x0, S0, C0)
    if not ok_sched:
        print("[HINT] LP→SSGS 仍不满足基本可行性，请先修这里（通常是 pred 或 eligible/p 的不一致）。")
    else:
        print()

    ok_ws = check_warm_feasibility(ops, eligible, proc_time, pred, machines, x0, S0, C0)
    # print(f"[DEBUG] ok_ws={ok_ws}")
    if ok_ws:
        UB_guess = max(Cmax_lb_init, int(math.ceil(UB_ws)))
        apply_warm_start(model, x0, S0, C0, UB=UB_guess)
        export_gantt(model, ops, eligible, out_dir=warm_dir)
        export_assignment_long(model, ops, eligible, out_dir=warm_dir)
        export_assignment_wide(model, ops, eligible, out_dir=warm_dir)
        print("[WARM-EXPORT] LP→SSGS 的排程 CSV 已输出到：", warm_dir.resolve())
        try:
            warm_start_adj(model, ops, eligible, proc_time)  # 若模型有 y/u
        except Exception as e:
            print("[WARN] warm_start_adj failed:", e)
    else:
        print("[INFO] LP-guided warm start invalid. Proceeding WITHOUT any warm start.")
    
    # === 调试：统计有值的整数变量 ===
    n_int = n_int_with_val = 0
    for v in model.component_data_objects(Var, descend_into=True):
        if v.is_binary() or v.is_integer():
            n_int += 1
            if v.value is not None:
                n_int_with_val += 1
    print(f"[DEBUG] ints total={n_int}, with_value={n_int_with_val}")

    # solve 前做一次度检查
    check_degree_balance(model)

    # 5) 导出 LP / MPS 到指定目录
    model.write(str(OUT_DIR / "model.mps"), format="mps")

    # 6）gurobi_persistent 两阶段求解
    solver = SolverFactory("gurobi_persistent")
    solver.set_instance(model)   # 把当前变量及其起点送入 Gurobi 内存

    # 日志与解文件
    solver.set_gurobi_param("LogFile", str(LOGS_DIR / f"{case_name}_gurobi.log"))
    solver.set_gurobi_param("ResultFile", str(OUT_DIR / "solution.sol"))
    
    solver.set_gurobi_param("Presolve",   2)
    solver.set_gurobi_param("Cuts",       2)
    solver.set_gurobi_param("Symmetry",   2)
    solver.set_gurobi_param("Threads",    16)
    solver.set_gurobi_param("NodefileStart", 0.5)

    # 显式 warm start
    if ok_ws:
        solver._warm_start()  
        grb = solver._solver_model
        print("[DEBUG] NumStart(after _warm_start) =", getattr(grb, "NumStart", -1))
        try:
            grb.write(str(OUT_DIR / "warmstart_debug.mst"))
            print("[DEBUG] wrote warmstart_debug.mst")
        except Exception as e:
            print("[WARN] write mst failed:", e)

    # ===== 兜底：若还是没有 MIP start，则手动把整型变量写入 Start =====
    if getattr(grb, "NumStart", 0) == 0:
        set_cnt = 0
        for v in model.component_data_objects(Var, descend_into=True):
            if not (v.is_binary() or v.is_integer()):
                continue
            if v.value is None:
                continue
            gv = grb.getVarByName(v.name)
            if gv is None:
                continue
            gv.Start = float(round(v.value))
            set_cnt += 1
        grb.update()
        print(f"[DEBUG] manual Starts set={set_cnt}, NumStart now={getattr(grb, 'NumStart', -1)}")
        try:
            grb.write(str(OUT_DIR / 'warmstart_debug.mst'))
            print("[DEBUG] wrote warmstart_debug.mst (manual)")
        except Exception as e:
            print("[WARN] write mst failed (manual):", e)

    # Phase-1：找可行解优先（180s）
    solver.set_gurobi_param("TimeLimit",  180)
    solver.set_gurobi_param("MIPFocus",   1)
    solver.set_gurobi_param("Heuristics", 0.8)
    solver.set_gurobi_param("PumpPasses", 50)
    solver.set_gurobi_param("RINS",       10)
    res1 = solver.solve(tee=True)
    solver.load_vars()  # 把 incumbent 写回 Pyomo 变量
    print("[INFO] Phase-1 incumbent Cmax =", pyo.value(model.Cmax))

    # Phase-2：若已有解则推界（420s + Cutoff），否则继续找解
    result_for_meta = res1
    if has_incumbent(res1, model):
        UB_curr = float(value(model.Cmax))
        solver.set_gurobi_param("Cutoff", UB_curr - 1e-6)
        solver.set_gurobi_param("TimeLimit", 420)
        solver.set_gurobi_param("MIPFocus",  3)
        solver.set_gurobi_param("Heuristics", 0.05)
        res2 = solver.solve(tee=True)
        result_for_meta = res2
    else:
        print("[WARN] No incumbent after phase-1. Staying in feasibility mode for another 420s.")
        solver.set_gurobi_param("TimeLimit", 420)
        solver.set_gurobi_param("MIPFocus",  1)
        solver.set_gurobi_param("Heuristics", 0.9)
        res2 = solver.solve(tee=True)
        result_for_meta = res2
    
    # 7) 导出数据以及可视化
    cmax_val = value(model.Cmax)
    print("Cmax =", cmax_val)
    
    # 只有有解时才导出甘特/赋值，避免误导
    if cmax_val is not None:
        export_gantt(model, ops, eligible, out_dir=OUT_DIR)
        export_assignment_long(model, ops, eligible, out_dir=OUT_DIR)
        export_assignment_wide(model, ops, eligible, out_dir=OUT_DIR)
    else:
        print("[WARN] No feasible solution found. Skipping schedule exports.")

    # 把本轮真实参数写进 meta（便于回溯）
    solver_opts_dump = {
        "phase1": {
            "TimeLimit": 180, "MIPFocus": 1, "Heuristics": 0.8,
            "PumpPasses": 50, "RINS": 10, "Presolve": 2, "Cuts": 2, "Symmetry": 2,
            "Threads": 16, "NodefileStart": 0.5,
        },
        "phase2": {
            "TimeLimit": 420,
            "MIPFocus": 3 if has_incumbent(res1) else 1,
            "Heuristics": 0.05 if has_incumbent(res1) else 0.9,
            "Cutoff": (float(cmax_val) - 1e-6) if (cmax_val is not None and has_incumbent(res1)) else None
        }
    }
    export_metadata(
        model, result_for_meta,
        out_dir=OUT_DIR,
        solver_name="gurobi_persistent",
        solver_options=solver_opts_dump,
        extra={
            "lp":  str((OUT_DIR/"model.lp").resolve()),
            "mps": str((OUT_DIR/"model.mps").resolve()),
            "sol": str((OUT_DIR/"solution.sol").resolve()),
            "log": str((LOGS_DIR/f"{case_name}_gurobi.log").resolve()),
            "lb_job": int(LB_job),
            "lb_assign": int(ceil(LB_assign)),
            "lb_init": int(Cmax_lb_init),
            "ub_safe": int(UB_safe),
            "warm_start_from_lp": bool(ok_ws),
        }
    )
    print("输出已导出到：", OUT_DIR.resolve(), LOGS_DIR.resolve())

if __name__ == "__main__":
    main()