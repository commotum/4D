# Vectorized NumPy version of the "Fast-scalar MonSTER Triad" (no Python loops over frequencies)
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) Minkowski helpers
# =============================================================================
ETA4 = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

def minkowski_dot_big_vec(u: np.ndarray, v: np.ndarray) -> float:
    """
    Vectorized big Minkowski inner product for (dim,) row vectors,
    where the metric is block-diagonal with copies of ETA4 on each 4-D chunk.
    """
    U = u.reshape(-1, 4)
    V = v.reshape(-1, 4)
    return np.sum((U @ ETA4) * V)


# =============================================================================
# 2) Vectorized scalar-table builder
# =============================================================================
SLICE = 12  # 12 dims per frequency: [X4 | Y4 | Z4]

class TriadMonSTERFastVec:
    """
    Vectorized cache of scalar tables for fast absolute/relative transforms.
    Frequencies: lam_j = base^{-j / F}, j=0..F-1, where F = dim // 12.

    Angles/Rapidities:
        phi_j   = (t * unit) * lam_j
        thx_j   = (x * unit) * lam_j
        thy_j   = (y * unit) * lam_j
        thz_j   = (z * unit) * lam_j

    Forward returns a dict with shapes:
        ch, sh:     (F,)          # cosh/sinh(phi)
        c_axes, s_axes: (F,3)     # cos/sin for X,Y,Z axes respectively
    """
    def __init__(self, dim: int = 768, base: float = 10000.0, top_delta: int = 1024):
        if dim % SLICE != 0:
            raise ValueError(f"dim must be divisible by {SLICE}, got {dim}.")
        self.dim      = dim
        self.num_freq = dim // SLICE
        self.base     = float(base)
        self.unit     = 10 / float(top_delta)  # global unit (dimensionless per step)
        j = np.arange(self.num_freq, dtype=np.float64)
        self.inv_freq = self.base ** (- j / self.num_freq)
        self._cache = {}

    def forward(self, s: np.ndarray):
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")
        key = (s[0], s[1], s[2], s[3], self.unit, self.base)
        if key in self._cache:
            return self._cache[key]

        t, x, y, z = s
        lam = self.inv_freq  # (F,)
        u = self.unit

        phi  = (t * u) * lam
        thx  = (x * u) * lam
        thy  = (y * u) * lam
        thz  = (z * u) * lam

        ch = np.cosh(phi)              # (F,)
        sh = np.sinh(phi)              # (F,)
        c_axes = np.stack((np.cos(thx), np.cos(thy), np.cos(thz)), axis=1)  # (F,3) -> X,Y,Z
        s_axes = np.stack((np.sin(thx), np.sin(thy), np.sin(thz)), axis=1)  # (F,3)

        out = {"ch": ch, "sh": sh, "c": c_axes, "s": s_axes}
        self._cache[key] = out
        return out


# =============================================================================
# 3) Vectorized apply (no loops over frequencies)
# =============================================================================
def apply_monster_triad_fast_vec(emb: np.ndarray, tables: dict, dim: int = 768) -> np.ndarray:
    """
    Apply triad transforms to a full embedding using only vectorized broadcasting.
    Args:
        emb   : (dim,) row vector.
        tables: dict with "ch","sh","c","s" from TriadMonSTERFastVec.forward.
        dim   : total embedding dimension (multiple of 12).
    Returns:
        (dim,) transformed row vector.
    """
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")
    F = dim // SLICE

    # Reshape into (F, 3, 4): freq buckets × {X,Y,Z} × [t,x,y,z]
    V = emb.reshape(F, 3, 4).astype(np.float64, copy=False)
    out = V.copy()

    # Broadcasted scalars
    ch = tables["ch"]          # (F,)
    sh = tables["sh"]          # (F,)
    c_axes = tables["c"]       # (F,3)
    s_axes = tables["s"]       # (F,3)

    # --------------------
    # Step 1: Boost along each axis' spatial component
    # --------------------
    # Indices for the spatial component aligned with axis: X->1, Y->2, Z->3
    comp_idx = np.array([1, 2, 3], dtype=np.int64)[None, :, None]  # (1,3,1)
    t = out[:, :, 0]                        # (F,3)
    x_axis = np.take_along_axis(out, comp_idx, axis=2)[..., 0]  # (F,3)

    t1 = ch[:, None] * t - sh[:, None] * x_axis
    x1 = -sh[:, None] * t + ch[:, None] * x_axis

    out[:, :, 0] = t1
    np.put_along_axis(out, comp_idx, x1[..., None], axis=2)

    # --------------------
    # Step 2: Rotate in the orthogonal spatial 2D planes
    # --------------------
    # For axis X: rotate (y,z) -> indices (2,3)
    # For axis Y: rotate (x,z) -> indices (1,3)
    # For axis Z: rotate (x,y) -> indices (1,2)
    pair_idx = np.array([[2, 3], [1, 3], [1, 2]], dtype=np.int64)[None, :, :]  # (1,3,2)

    pair_vals = np.take_along_axis(out, pair_idx, axis=2)  # (F,3,2)
    u = pair_vals[..., 0]  # first in the pair
    v = pair_vals[..., 1]  # second in the pair

    cu = c_axes  # (F,3)
    su = s_axes  # (F,3)

    u2 = cu * u - su * v
    v2 = su * u + cu * v

    rotated = np.stack((u2, v2), axis=-1)  # (F,3,2)
    np.put_along_axis(out, pair_idx, rotated, axis=2)

    return out.reshape(dim,)

def report_remote_attenuation(monster, dim, distances=None, num_runs=128, seed=1):
    print("\nRemote attenuation (identical embeddings)")
    if distances is None:
        distances = [0, 8, 16, 32, 64, 128, 256, 512]
    distances = np.array(distances, dtype=np.float64)
    direction = np.array([1.0, 0.6, -0.4, 0.2], dtype=np.float64)

    tables = [monster.forward(direction * d) for d in distances]
    rng = np.random.default_rng(seed)

    avg_abs = np.zeros(distances.size, dtype=np.float64)
    used = 0
    for _ in range(num_runs):
        v = rng.uniform(-0.6, 0.6, size=dim).astype(np.float64)
        vv = minkowski_dot_big_vec(v, v)
        denom = abs(vv)
        if denom < 1e-12:
            continue
        used += 1
        inv_denom = 1.0 / denom
        for j, T in enumerate(tables):
            v_rel = apply_monster_triad_fast_vec(v, T, dim=dim)
            dot = minkowski_dot_big_vec(v, v_rel)
            avg_abs[j] += abs(dot) * inv_denom

    if used == 0:
        print("No valid samples (near-lightlike vectors).")
        return

    avg_abs /= used
    print(f"direction (t,x,y,z): {direction}")
    print(f"samples: {used} | distances: {distances.astype(int).tolist()}")
    print("avg abs similarity = |<v, L(d)v>| / |<v,v>|")
    print("distance | avg | bar")
    max_abs = max(1e-12, np.max(avg_abs))
    for d, s in zip(distances, avg_abs):
        bar = "#" * int((abs(s) / max_abs) * 20)
        print(f"{int(d):>8} | {s:>8.5f} | {bar}")

    nonzero = distances > 0
    if np.any(nonzero):
        slope = np.polyfit(np.log10(distances[nonzero]), avg_abs[nonzero], 1)[0]
        first_nz = int(np.flatnonzero(nonzero)[0])
        tail_smaller = avg_abs[-1] < avg_abs[first_nz]
        print(f"trend slope (log10 dist vs avg abs sim): {slope:+.6f}")
        print(f"attenuation observed? {tail_smaller and slope < 0}")

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)

def build_permutation_table(spatial_radii, temporal_radii, mixed_temporal):
    rows = []

    def add_row(dt, dx, dy, dz, category):
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        s2 = dt * dt - r * r
        if abs(s2) < 1e-9:
            interval_class = "null"
        elif s2 > 0:
            interval_class = "timelike"
        else:
            interval_class = "spacelike"
        if dt > 0:
            dt_dir = "future"
        elif dt < 0:
            dt_dir = "past"
        else:
            dt_dir = "simultaneous"

        axis = ""
        if dx != 0 and dy == 0 and dz == 0:
            axis = "x"
        elif dy != 0 and dx == 0 and dz == 0:
            axis = "y"
        elif dz != 0 and dx == 0 and dy == 0:
            axis = "z"

        r_mult = ""
        if dt != 0 and r != 0:
            r_mult = r / abs(dt)

        rows.append(
            {
                "category": category,
                "dt": float(dt),
                "dx": float(dx),
                "dy": float(dy),
                "dz": float(dz),
                "dt_abs": float(abs(dt)),
                "r": float(r),
                "s2": float(s2),
                "interval_class": interval_class,
                "dt_dir": dt_dir,
                "axis": axis,
                "r_mult": r_mult,
            }
        )

    add_row(0.0, 0.0, 0.0, 0.0, "baseline")

    for r in spatial_radii:
        for axis in ("x", "y", "z"):
            for sign in (-1.0, 1.0):
                dx = dy = dz = 0.0
                if axis == "x":
                    dx = sign * r
                elif axis == "y":
                    dy = sign * r
                else:
                    dz = sign * r
                add_row(0.0, dx, dy, dz, "space_only")

    for t in temporal_radii:
        for sign in (-1.0, 1.0):
            add_row(sign * t, 0.0, 0.0, 0.0, "time_only")

    for t in mixed_temporal:
        for sign in (-1.0, 1.0):
            dt = sign * abs(t)
            for r_mult in (0.5, 1.0, 2.0):
                r = r_mult * abs(t)
                for axis in ("x", "y", "z"):
                    dx = dy = dz = 0.0
                    if axis == "x":
                        dx = r
                    elif axis == "y":
                        dy = r
                    else:
                        dz = r
                    add_row(dt, dx, dy, dz, "mixed")

    for i, row in enumerate(rows):
        row["row_id"] = i
    return rows

def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

def compute_attention_stats(monster, dim, rows, num_runs=128, seed=0, w_seed=123):
    rng = np.random.default_rng(seed)
    rng_w = np.random.default_rng(w_seed)
    Wq = rng_w.normal(1.0, 0.02, size=dim).astype(np.float64)
    Wk = rng_w.normal(1.0, 0.02, size=dim).astype(np.float64)

    X = rng.uniform(-0.6, 0.6, size=(num_runs, dim)).astype(np.float64)
    Q0 = X * Wq
    K0 = X * Wk

    l_AA_samples = np.array([minkowski_dot_big_vec(Q0[i], K0[i]) for i in range(num_runs)], dtype=np.float64)
    mean_l_AA = float(np.mean(l_AA_samples))

    out_rows = []
    for row in rows:
        s = np.array([row["dt"], row["dx"], row["dy"], row["dz"]], dtype=np.float64)
        T_B = monster.forward(s)

        sum_l_AB = 0.0
        sum_l_BA = 0.0
        sum_l_BB = 0.0
        sum_gap_A = 0.0
        sum_gap_B = 0.0
        sum_p_A_B = 0.0
        sum_p_B_A = 0.0
        cnt_AB_lt_AA = 0
        cnt_BA_lt_BB = 0
        cnt_AB_lt_0 = 0
        cnt_BA_lt_0 = 0

        for i in range(num_runs):
            qA = Q0[i]
            kA = K0[i]
            l_AA = l_AA_samples[i]

            qB = apply_monster_triad_fast_vec(qA, T_B, dim=dim)
            kB = apply_monster_triad_fast_vec(kA, T_B, dim=dim)

            l_AB = minkowski_dot_big_vec(qA, kB)
            l_BA = minkowski_dot_big_vec(qB, kA)
            l_BB = minkowski_dot_big_vec(qB, kB)

            gap_A = l_AB - l_AA
            gap_B = l_BA - l_BB
            p_A_B = sigmoid(gap_A)
            p_B_A = sigmoid(gap_B)

            sum_l_AB += l_AB
            sum_l_BA += l_BA
            sum_l_BB += l_BB
            sum_gap_A += gap_A
            sum_gap_B += gap_B
            sum_p_A_B += p_A_B
            sum_p_B_A += p_B_A
            cnt_AB_lt_AA += int(l_AB < l_AA)
            cnt_BA_lt_BB += int(l_BA < l_BB)
            cnt_AB_lt_0 += int(l_AB < 0.0)
            cnt_BA_lt_0 += int(l_BA < 0.0)

        inv = 1.0 / num_runs
        out = dict(row)
        out.update(
            {
                "mean_l_AA": mean_l_AA,
                "mean_l_AB": sum_l_AB * inv,
                "mean_l_BA": sum_l_BA * inv,
                "mean_l_BB": sum_l_BB * inv,
                "mean_gap_A": sum_gap_A * inv,
                "mean_gap_B": sum_gap_B * inv,
                "mean_p_A_B": sum_p_A_B * inv,
                "mean_p_B_A": sum_p_B_A * inv,
                "frac_AB_lt_AA": cnt_AB_lt_AA * inv,
                "frac_BA_lt_BB": cnt_BA_lt_BB * inv,
                "frac_AB_lt_0": cnt_AB_lt_0 * inv,
                "frac_BA_lt_0": cnt_BA_lt_0 * inv,
            }
        )
        out_rows.append(out)

    return out_rows

def fmt_tick(x):
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"

def plot_time_only(stats_rows, out_path: Path):
    rows = [r for r in stats_rows if r["r"] < 1e-9 and r["category"] in ("baseline", "time_only")]
    baseline = [r for r in rows if abs(r["dt"]) < 1e-9]
    base_val = baseline[0]["mean_p_A_B"] if baseline else None

    future = sorted([r for r in rows if r["dt"] > 0], key=lambda r: r["dt_abs"])
    past = sorted([r for r in rows if r["dt"] < 0], key=lambda r: r["dt_abs"])

    x_f = [r["dt_abs"] for r in future]
    y_f = [r["mean_p_A_B"] for r in future]
    x_p = [r["dt_abs"] for r in past]
    y_p = [r["mean_p_A_B"] for r in past]

    if base_val is not None:
        x_f = [0.0] + x_f
        y_f = [base_val] + y_f
        x_p = [0.0] + x_p
        y_p = [base_val] + y_p

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_f, y_f, marker="o", label="future dt")
    ax.plot(x_p, y_p, marker="o", label="past dt")
    ax.set_xlabel("abs(dt)")
    ax.set_ylabel("mean p_A_to_B")
    ax.set_title("Time-only attenuation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_space_only(stats_rows, out_path: Path):
    rows = [r for r in stats_rows if r["category"] == "space_only"]
    axis_map = {"x": {}, "y": {}, "z": {}}
    for r in rows:
        axis = r["axis"]
        if axis not in axis_map:
            continue
        rad = r["r"]
        axis_map[axis].setdefault(rad, []).append(r["mean_p_A_B"])

    fig, ax = plt.subplots(figsize=(6, 4))
    for axis, r_map in axis_map.items():
        if not r_map:
            continue
        r_vals = sorted(r_map.keys())
        y_vals = [float(np.mean(r_map[v])) for v in r_vals]
        ax.plot(r_vals, y_vals, marker="o", label=f"{axis}-axis")

    ax.set_xlabel("r (spatial radius)")
    ax.set_ylabel("mean p_A_to_B")
    ax.set_title("Space-only attenuation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_mixed_heatmap(stats_rows, out_path: Path):
    rows = [r for r in stats_rows if r["category"] == "mixed"]
    if not rows:
        return

    dt_vals = sorted({r["dt_abs"] for r in rows})
    r_vals = sorted({r["r"] for r in rows})
    grid = np.full((len(dt_vals), len(r_vals)), np.nan, dtype=np.float64)
    counts = np.zeros_like(grid)
    class_grid = [["" for _ in r_vals] for _ in dt_vals]

    dt_index = {v: i for i, v in enumerate(dt_vals)}
    r_index = {v: j for j, v in enumerate(r_vals)}

    for r in rows:
        i = dt_index[r["dt_abs"]]
        j = r_index[r["r"]]
        if np.isnan(grid[i, j]):
            grid[i, j] = 0.0
        grid[i, j] += r["mean_p_A_B"]
        counts[i, j] += 1.0
        class_grid[i][j] = r["interval_class"]

    grid = np.divide(grid, np.where(counts > 0, counts, 1.0))
    masked = np.ma.masked_invalid(grid)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(r_vals)))
    ax.set_xticklabels([fmt_tick(v) for v in r_vals])
    ax.set_yticks(np.arange(len(dt_vals)))
    ax.set_yticklabels([fmt_tick(v) for v in dt_vals])
    ax.set_xlabel("r (spatial radius)")
    ax.set_ylabel("abs(dt)")
    ax.set_title("Mixed spacetime attenuation")
    fig.colorbar(im, ax=ax, label="mean p_A_to_B")

    for i in range(len(dt_vals)):
        for j in range(len(r_vals)):
            if counts[i, j] <= 0:
                continue
            cls = class_grid[i][j]
            if not cls:
                continue
            letter = {"timelike": "T", "null": "N", "spacelike": "S"}.get(cls, "?")
            ax.text(j, i, letter, ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def run_permutation_demo(monster, dim, num_runs=128, seed=0, w_seed=123, output_dir=None):
    spatial_radii = [8, 16, 32, 64, 128, 256]
    temporal_radii = [8, 16, 32, 64, 128, 256]
    mixed_temporal = [64, 128, 256]

    rows = build_permutation_table(spatial_radii, temporal_radii, mixed_temporal)
    stats_rows = compute_attention_stats(monster, dim, rows, num_runs=num_runs, seed=seed, w_seed=w_seed)

    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "v13_proof_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "monster_spacetime_permutation_table.csv"
    stats_path = out_dir / "monster_attention_demo_stats.csv"
    plot_time = out_dir / "monster_time_only.png"
    plot_space = out_dir / "monster_space_only.png"
    plot_mixed = out_dir / "monster_mixed_heatmap.png"

    table_fields = [
        "row_id",
        "category",
        "dt",
        "dx",
        "dy",
        "dz",
        "dt_abs",
        "r",
        "s2",
        "interval_class",
        "dt_dir",
        "axis",
        "r_mult",
    ]
    stats_fields = table_fields + [
        "mean_l_AA",
        "mean_l_AB",
        "mean_l_BA",
        "mean_l_BB",
        "mean_gap_A",
        "mean_gap_B",
        "mean_p_A_B",
        "mean_p_B_A",
        "frac_AB_lt_AA",
        "frac_BA_lt_BB",
        "frac_AB_lt_0",
        "frac_BA_lt_0",
    ]

    write_csv(table_path, rows, table_fields)
    write_csv(stats_path, stats_rows, stats_fields)

    plot_time_only(stats_rows, plot_time)
    plot_space_only(stats_rows, plot_space)
    plot_mixed_heatmap(stats_rows, plot_mixed)

    print(f"\nProof demo outputs written to: {out_dir}")
    print(f"  permutation table: {table_path.name}")
    print(f"  stats table      : {stats_path.name}")
    print(f"  time-only plot   : {plot_time.name}")
    print(f"  space-only plot  : {plot_space.name}")
    print(f"  mixed heatmap    : {plot_mixed.name}")


# =============================================================================
# 4) Demo / Sanity checks
# =============================================================================
if __name__ == "__main__":
    np.random.seed(0)
    DIM = 768
    monster = TriadMonSTERFastVec(dim=DIM, base=10000.0, top_delta=1024)

    # Absolute 4D positions (in "steps")
    s_q  = np.array([ 700.0,  500.0, -300.0,  200.0], dtype=np.float64)  # (t,x,y,z)
    s_k  = np.array([ -40.0,  -20.0,   60.0,  -10.0], dtype=np.float64)
    dskq = s_k - s_q

    # Tables
    T_abs_q = monster.forward(s_q)
    T_abs_k = monster.forward(s_k)
    T_rel   = monster.forward(dskq)

    # Random embeddings
    q = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)
    k = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)

    # Apply absolute maps
    q_abs = apply_monster_triad_fast_vec(q, T_abs_q, dim=DIM)
    k_abs = apply_monster_triad_fast_vec(k, T_abs_k, dim=DIM)

    # RoPE-style identity check
    lhs = minkowski_dot_big_vec(q_abs, k_abs)
    k_rel = apply_monster_triad_fast_vec(k, T_rel, dim=DIM)
    rhs = minkowski_dot_big_vec(q, k_rel)

    print("RoPE-style identity holds? ", np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12))
    print(f"lhs: {lhs:+.12f}  rhs: {rhs:+.12f}")

    # Per-4D Minkowski norm preservation
    Q_blocks = q.reshape(-1, 4)
    Q_abs_blocks = q_abs.reshape(-1, 4)
    norms_before = np.sum((Q_blocks @ ETA4) * Q_blocks, axis=1)
    norms_after  = np.sum((Q_abs_blocks @ ETA4) * Q_abs_blocks, axis=1)
    ok = np.allclose(norms_before, norms_after, rtol=1e-11, atol=1e-12)
    max_err = np.max(np.abs(norms_before - norms_after))
    print("Per-4D Minkowski norms preserved? ", ok, "| max abs err:", max_err)

    report_remote_attenuation(monster, DIM)
    run_permutation_demo(monster, DIM)

    print("NUM_FREQ:", DIM // SLICE, " | DIM:", DIM, " | SLICE per freq:", SLICE)
