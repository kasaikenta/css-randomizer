# -*- coding: utf-8 -*-
"""
CSS_randomizer.py
  - Sparse (SciPy CSR) implementation for constructing random orthogonal
    sparse matrix pairs (HX, HZ) suitable for CSS/quantum LDPC codes.
  - Applies one random 2x2 cross-switch to HX per iteration.
  - Builds a reduced GF(2) linear system only on violated rows/columns (I, J, K).
  - Repairs HZ via a CP-SAT ILP that preserves row/column weights and restores orthogonality.
  - Uses sparse_xor_mod2() for XOR (mod 2) on sparse matrices.
  - Normalizes getnnz(axis=...) comparisons via nnz_vec().
"""

import numpy as np
from scipy import sparse as sp
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# =========================
# Helper utilities
# =========================
def sparse_xor_mod2(A: sp.spmatrix, B: sp.spmatrix) -> sp.csr_matrix:
    """
    Elementwise XOR (addition mod 2) for sparse matrices.

    Args:
        A: Sparse matrix (any SciPy sparse type), entries assumed in {0,1}.
        B: Sparse matrix of the same shape, entries in {0,1}.

    Returns:
        CSR sparse matrix of dtype uint8 representing (A XOR B) over GF(2).
    """
    return (A.astype(bool) != B.astype(bool)).astype(np.uint8).tocsr()

def nnz_vec(A: sp.spmatrix, axis: int) -> np.ndarray:
    """
    Returns the per-row (axis=1) or per-column (axis=0) nonzero counts
    as a 1-D NumPy array.

    Args:
        A: Sparse matrix.
        axis: 0 for column counts, 1 for row counts.

    Returns:
        1-D ndarray of int counts along the specified axis.
    """
    return np.asarray(A.getnnz(axis=axis)).ravel()

# =========================
# Core utilities (sparse)
# =========================
def perm_from_affine(a: int, b: int, P: int) -> np.ndarray:
    """
    Builds a permutation x -> a*x + b (mod P).

    Args:
        a: Multiplicative coefficient (invertible mod P).
        b: Additive coefficient.
        P: Modulus.

    Returns:
        1-D ndarray of length P with permuted indices.
    """
    return np.array([(a*int(x) + b) % P for x in range(P)], dtype=int)

def block_perm_matrix(a: int, b: int, P: int) -> sp.csr_matrix:
    """
    Creates a P×P permutation matrix corresponding to x -> a*x + b (mod P).

    Args:
        a, b, P: As above.

    Returns:
        CSR sparse permutation matrix of shape (P, P).
    """
    perm = perm_from_affine(a, b, P)
    rows = np.arange(P, dtype=int)
    cols = perm
    data = np.ones(P, dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=(P, P))

def inv_affine(a: int, b: int, P: int) -> tuple[int, int]:
    """
    Returns inverse affine parameters for x -> a*x + b (mod P).

    Args:
        a: Multiplicative coefficient (invertible mod P).
        b: Additive coefficient.
        P: Modulus.

    Returns:
        (a_inv, b_inv) such that a_inv*(a*x + b) + b_inv ≡ x (mod P).
    """
    a_inv = pow(int(a), -1, P)
    b_inv = (-a_inv * b) % P
    return a_inv, b_inv

def _bmat(table, fmt='csr') -> sp.csr_matrix:
    """
    Thin wrapper around scipy.sparse.bmat.

    Args:
        table: 2-D list of sparse blocks.
        fmt: Output sparse format.

    Returns:
        Block matrix in the requested sparse format.
    """
    return sp.bmat(table, format=fmt)

def make_hat_matrices(dc: int, f_list, g_list, f_inv, g_inv, P: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Builds (HX, HZ) block-circulant matrices from affine permutation blocks.

    Args:
        dc: Number of block-rows (left/right each is an n×n block-matrix).
        f_list, g_list: Lists of (a,b) for affine permutation matrices.
        f_inv, g_inv: Lists of inverse (a,b) for f_list, g_list.
        P: Inner block size.

    Returns:
        (HX, HZ) as CSR sparse matrices.
    """
    n = len(f_list)
    HX_L = [[f_list[(l - j) % n] for l in range(n)] for j in range(n)]
    HX_R = [[g_list[(l - j) % n] for l in range(n)] for j in range(n)]
    HZ_L = [[g_inv[(-(l - j)) % n] for l in range(n)] for j in range(n)]
    HZ_R = [[f_inv[(-(l - j)) % n] for l in range(n)] for j in range(n)]

    def make_block_matrix(tbl):
        blocks = [[block_perm_matrix(a, b, P) for (a, b) in row] for row in tbl]
        return _bmat(blocks, fmt='csr')

    HX = _bmat([[make_block_matrix(HX_L), make_block_matrix(HX_R)]], fmt='csr')
    HZ = _bmat([[make_block_matrix(HZ_L), make_block_matrix(HZ_R)]], fmt='csr')
    return HX, HZ

def make_affine_pair(P=5, dc=2, nf=3, ng=3, seed=0) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Randomly samples affine parameters and constructs an (HX, HZ) pair.

    Args:
        P: Prime modulus for inner permutation blocks.
        dc: Not used directly here; kept for API compatibility.
        nf, ng: Number of f and g affine blocks.
        seed: RNG seed.

    Returns:
        (HX, HZ) as CSR sparse matrices.
    """
    rng = np.random.default_rng(seed)
    a_f = rng.choice([1, P-1], nf)
    b_f = rng.integers(0, P, nf)

    a_g, b_g = [], []
    for _ in range(ng):
        while True:
            c = rng.choice([1, P-1])
            d = rng.integers(0, P)
            ok = True
            for j in range(nf):
                if (a_f[j]-1)*d % P != (c-1)*b_f[j] % P:
                    ok = False
                    break
            if ok:
                a_g.append(c); b_g.append(d); break

    f_list = list(zip(a_f, b_f))
    g_list = list(zip(a_g, b_g))
    f_inv = [inv_affine(a, b, P) for a, b in f_list]
    g_inv = [inv_affine(a, b, P) for a, b in g_list]
    HX, HZ = make_hat_matrices(dc, f_list, g_list, f_inv, g_inv, P)
    return HX, HZ

# ==== GF(2) ====
def gf2_matmul(A: sp.spmatrix, B: sp.spmatrix) -> sp.csr_matrix:
    """
    Sparse matrix product over GF(2).

    Args:
        A, B: Sparse matrices with entries in {0,1}.

    Returns:
        CSR matrix representing (A @ B) mod 2.
    """
    C = (A @ B).tocsr(copy=True)
    if C.nnz:
        C.data %= 2
        C.eliminate_zeros()
    return C

def compute_syndrome(HX: sp.spmatrix, HZ: sp.spmatrix) -> sp.csr_matrix:
    """
    Computes the orthogonality syndrome HX * HZ^T over GF(2).

    Args:
        HX, HZ: Sparse matrices with entries in {0,1}.

    Returns:
        CSR matrix of the syndrome.
    """
    return gf2_matmul(HX, HZ.transpose())

def make_block_id_matrix(P: int, n_block_rows=2, n_block_cols=6) -> sp.csr_matrix:
    """
    Constructs a tiled block identity: kron(ones, I_P).

    Args:
        P: Inner identity block size.
        n_block_rows, n_block_cols: Tiling factors.

    Returns:
        CSR sparse matrix of shape (n_block_rows*P, n_block_cols*P).
    """
    I = sp.eye(P, dtype=np.uint8, format='csr')
    ones = sp.csr_matrix(np.ones((n_block_rows, n_block_cols), dtype=np.uint8))
    return sp.kron(ones, I, format='csr')

# ==== 2×2 switch (sparse) ====
def random_cross_swap_sparse(H: sp.csr_matrix, max_trials=2000):
    """
    Performs one random 2x2 cross-switch on H if possible.

    Pattern:
        [1 0]      [0 1]
        [0 1]  or  [1 0]
    is flipped, preserving row/column weights.

    Args:
        H: CSR matrix with entries in {0,1}.
        max_trials: Max random attempts to find a valid 2x2 switch.

    Returns:
        (H_switched, (r1, r2, c1, c2)) on success, (None, None) if not found.
    """
    rows, cols = H.nonzero()
    nnz = len(rows)
    if nnz < 2:
        return None, None
    Hd = H.todok(copy=True)
    for _ in range(max_trials):
        idx1 = rng.integers(nnz)
        idx2 = rng.integers(nnz)
        r1, c1 = int(rows[idx1]), int(cols[idx1])
        r2, c2 = int(rows[idx2]), int(cols[idx2])
        if r1 == r2 or c1 == c2:
            continue
        a11 = 1 if (r1, c1) in Hd else 0
        a22 = 1 if (r2, c2) in Hd else 0
        a12 = 1 if (r1, c2) in Hd else 0
        a21 = 1 if (r2, c1) in Hd else 0
        if a11 and a22 and (not a12) and (not a21):
            Hd[r1, c1] = 0; Hd[r2, c2] = 0
            Hd[r1, c2] = 1; Hd[r2, c1] = 1
            return Hd.tocsr(), (r1, r2, c1, c2)
        if a12 and a21 and (not a11) and (not a22):
            Hd[r1, c2] = 0; Hd[r2, c1] = 0
            Hd[r1, c1] = 1; Hd[r2, c2] = 1
            return Hd.tocsr(), (r1, r2, c1, c2)
    return None, None

# ==== I, J, K ====
def extract_I_J_K(HXp: sp.csr_matrix, HZ: sp.csr_matrix):
    """
    Extracts index sets I, J, K based on the current syndrome.

    I: indices of rows in HZ where orthogonality is violated (nonzero columns in S).
    J: columns used by HZ[I, :].
    K: rows of HX that touch columns J.

    Args:
        HXp: HX after a tentative switch (CSR).
        HZ: Current HZ (CSR).

    Returns:
        (I, J, K, S) where S = HX*HZ^T (CSR).
    """
    S = compute_syndrome(HXp, HZ)
    I = np.where(S.getnnz(axis=0) > 0)[0]
    if len(I) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), S
    HZ_I = HZ[I, :]
    J = np.where(HZ_I.getnnz(axis=0) > 0)[0]
    A = HXp[:, J]
    K = np.where(A.getnnz(axis=1) > 0)[0]
    return I, J, K, S

# ==== GF(2) solver (dense small systems) ====
def gf2_solve_all_solutions(A: np.ndarray, b: np.ndarray):
    """
    Solves A x = b over GF(2) and returns a particular solution and a nullspace basis.

    Args:
        A: (m, n) uint8 matrix over GF(2).
        b: (m,) or (m, k) uint8 right-hand side over GF(2).

    Returns:
        (x0, Z, piv, free) where
          - x0 is a particular solution (n,) or (n, k),
          - Z is a (n, d) basis for the nullspace,
          - piv is the list of pivot column indices,
          - free is the list of free column indices.
        Returns (None, None, None, None) if infeasible.
    """
    A = (A & 1).astype(np.uint8); b = (b & 1).astype(np.uint8)
    m, n = A.shape
    if b.ndim == 1:
        b = b.reshape(m, 1)
    Ab = np.concatenate([A.copy(), b.copy()], axis=1).astype(np.uint8)
    rows, cols = m, n
    r = 0; piv = []
    for c in range(cols):
        p = None
        for rr in range(r, rows):
            if Ab[rr, c]:
                p = rr; break
        if p is None:
            continue
        if p != r:
            Ab[[r, p], :] = Ab[[p, r], :]
        for rr in range(rows):
            if rr != r and Ab[rr, c]:
                Ab[rr, :] ^= Ab[r, :]
        piv.append(c); r += 1
        if r == rows:
            break
    for rr in range(rows):
        if not Ab[rr, :cols].any() and Ab[rr, cols:].any():
            return None, None, None, None
    free = [j for j in range(cols) if j not in piv]
    k_rhs = Ab.shape[1] - cols
    x0 = np.zeros((cols, k_rhs), dtype=np.uint8)
    for ri in reversed(range(len(piv))):
        c = piv[ri]; s = Ab[ri, cols:].copy()
        for cc in range(c+1, cols):
            if Ab[ri, cc]:
                s ^= x0[cc, :]
        x0[c, :] = s
    if k_rhs == 1:
        x0 = x0[:, 0]
    Z = np.zeros((cols, len(free)), dtype=np.uint8)
    for j_idx, j in enumerate(free):
        v = np.zeros(cols, dtype=np.uint8); v[j] = 1
        for ri in reversed(range(len(piv))):
            c = piv[ri]; s = 0
            for cc in range(c+1, cols):
                if Ab[ri, cc] and v[cc]:
                    s ^= 1
            v[c] = s
        Z[:, j_idx] = v
    return x0, Z, piv, free

# ==== GF(2) rank ====
def gf2_rank(A: np.ndarray) -> int:
    """
    Returns rank of a GF(2) matrix via elimination.

    Args:
        A: (m, n) uint8 matrix over GF(2).

    Returns:
        Rank as an integer.
    """
    M = (A.copy() & 1).astype(np.uint8)
    m, n = M.shape; r = 0
    for c in range(n):
        piv = None
        for rr in range(r, m):
            if M[rr, c]:
                piv = rr; break
        if piv is None:
            continue
        if piv != r:
            M[[r, piv], :] = M[[piv, r], :]
        for rr in range(m):
            if rr != r and M[rr, c]:
                M[rr, :] ^= M[r, :]
        r += 1
        if r == m:
            break
    return r

def estimate_degrees_of_freedom(HXp: sp.csr_matrix, HZ: sp.csr_matrix,
                                add_parity: bool = True, verbose: bool = True):
    """
    Estimates degrees of freedom d ≈ |I||J| - rank(A_big) for the reduced system.

    Args:
        HXp: HX after random switch.
        HZ: Current HZ.
        add_parity: Whether to add row/column parity constraints.
        verbose: Print summary.

    Returns:
        (d_est, total_est, I, J, K) where total_est ≈ 2^d (may be 'inf').
    """
    I, J, K, S = extract_I_J_K(HXp, HZ)
    if len(I) == 0:
        if verbose:
            print("[estimate] S = 0 → d = 0, total = 1")
        return 0, 1, I, J, K
    A_big, b_big, I, J, K, S, idx = build_system_with_K(HXp, HZ, add_parity=add_parity)
    v = len(I) * len(J)
    rank_big = gf2_rank(A_big)
    d = max(0, v - rank_big)
    total_est = 1 << d if d <= 60 else float("inf")
    if verbose:
        print(f"[estimate] |I|={len(I)}, |J|={len(J)}, |K|={len(K)}")
        print(f"[estimate] rank(A_big)={rank_big} / vars={v} → d={d}")
        if np.isfinite(total_est):
            print(f"[estimate] approx solutions ≈ 2^{d}")
        else:
            print(f"[estimate] approx solutions ≈ 2^{d} (huge)")
    return d, total_est, I, J, K

# ==== Build reduced system with K (sparse→small dense) ====
def build_system_with_K(HXp: sp.csr_matrix, HZ: sp.csr_matrix, add_parity: bool = False):
    """
    Builds the reduced GF(2) system A_big x = b_big using sets I, J, K.

    Args:
        HXp: HX after random switch (CSR).
        HZ: Current HZ (CSR).
        add_parity: If True, also add row/column parity-preservation equations.

    Returns:
        (A_big, b_big, I, J, K, S_full, idx)
        where A_big, b_big are dense uint8 arrays; idx maps (i, j) to variable index.
        If no violation, returns (None, None, I, J, K, S, {}).
    """
    I, J, K, S = extract_I_J_K(HXp, HZ)
    if len(I) == 0:
        return None, None, I, J, K, S, {}
    Ii, Jj, Kk = list(I), list(J), list(K)
    v = len(Ii) * len(Jj)
    idx = {(ii, jj): t for t, (ii, jj) in enumerate(((ii, jj) for ii in Ii for jj in Jj))}
    Ared = (HXp[Kk, :][:, Jj]).astype(np.uint8).toarray()      # |K|×|J|
    S_full = compute_syndrome(HXp, HZ)                          # m×n
    Bred = (S_full[Kk, :][:, Ii]).astype(np.uint8).toarray()    # |K|×|I|
    rows_A = []; rows_b = []
    for i_pos, i in enumerate(Ii):
        s_i = Bred[:, i_pos]
        for r in range(Ared.shape[0]):
            row = np.zeros(v, dtype=np.uint8)
            nz = np.nonzero(Ared[r, :])[0]
            for j_pos in nz:
                j = Jj[j_pos]
                row[idx[(i, j)]] ^= 1
            rows_A.append(row); rows_b.append(np.uint8(s_i[r]))
    if add_parity:
        # Row parity
        for i in Ii:
            row = np.zeros(v, dtype=np.uint8)
            for j in Jj:
                row[idx[(i, j)]] ^= 1
            rows_A.append(row); rows_b.append(np.uint8(0))
        # Column parity
        for j in Jj:
            row = np.zeros(v, dtype=np.uint8)
            for i in Ii:
                row[idx[(i, j)]] ^= 1
            rows_A.append(row); rows_b.append(np.uint8(0))
    A_big = np.stack(rows_A, axis=0).astype(np.uint8)
    b_big = np.array(rows_b, dtype=np.uint8)
    return A_big, b_big, I, J, K, S_full, idx

# ==== ILP repair with weight preservation ====
def solve_weight_preserving_ilp(K_pack,
                                HXp: sp.csr_matrix,
                                HZ: sp.csr_matrix,
                                time_limit_s: float = 10,
                                minimize_flips: bool = True):
    """
    Solves for Δ on (I, J) such that:
      (1) A_big * vec(Δ) ≡ b_big (mod 2),
      (2) Row/column signed-balance constraints preserve row/col weights,
      (3) Objective optionally minimizes the number of flips.

    Args:
        K_pack: Output of build_system_with_K(...).
        HXp: HX after random switch.
        HZ: Current HZ.
        time_limit_s: CP-SAT time limit in seconds.
        minimize_flips: If True, minimize the Hamming weight of Δ.

    Returns:
        (found, HZp, Delta) where HZp = HZ XOR Δ, or (False, None, None) if infeasible.
    """
    A_big, b_big, I, J, K, S, idx = K_pack
    if A_big is None:
        # No violation
        return True, HZ.copy(), sp.csr_matrix(HZ.shape, dtype=np.uint8)

    Ii, Jj = list(I), list(J)
    v = len(Ii) * len(Jj)

    # Dense sub-block of HZ over I×J (for signed-balance coefficients).
    HZ_IJ = (HZ[Ii, :][:, Jj]).astype(np.uint8).toarray()  # |I|×|J|

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{k}") for k in range(v)]

    # GF(2) orthogonality constraints: A_big x - 2 s = b_big
    s = [model.NewIntVar(0, A_big.shape[1]//2 + 1, f"s_{r}") for r in range(A_big.shape[0])]
    for r in range(A_big.shape[0]):
        terms = [x[k] for k in np.nonzero(A_big[r])[0]]
        lhs = cp_model.LinearExpr.Sum(terms)
        model.Add(lhs - 2 * s[r] == int(b_big[r]))

    # Row signed-balance on I
    Jlen = len(Jj)
    for ii in range(len(Ii)):
        coeffs = []; vars_ = []
        for jj in range(Jlen):
            k = ii * Jlen + jj
            c = 1 - 2*int(HZ_IJ[ii, jj])  # 1→-1, 0→+1
            if c != 0:
                coeffs.append(c); vars_.append(x[k])
        if vars_:
            model.Add(cp_model.LinearExpr.WeightedSum(vars_, coeffs) == 0)

    # Column signed-balance on J
    for jj in range(Jlen):
        coeffs = []; vars_ = []
        for ii in range(len(Ii)):
            k = ii * Jlen + jj
            c = 1 - 2*int(HZ_IJ[ii, jj])
            if c != 0:
                coeffs.append(c); vars_.append(x[k])
        if vars_:
            model.Add(cp_model.LinearExpr.WeightedSum(vars_, coeffs) == 0)

    if minimize_flips:
        model.Minimize(cp_model.LinearExpr.Sum(x))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.cp_model_presolve = True

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False, None, None

    # Reconstruct Δ (sparse)
    Delta_dok = sp.dok_matrix(HZ.shape, dtype=np.uint8)
    t = 0
    for ii, i in enumerate(Ii):
        for jj, j in enumerate(Jj):
            val = int(solver.Value(x[t])) & 1
            if val:
                Delta_dok[i, j] = 1
            t += 1
    Delta = Delta_dok.tocsr()
    HZp = sparse_xor_mod2(HZ, Delta)

    # Safety checks
    if compute_syndrome(HXp, HZp).nnz != 0:
        return False, None, None
    if not np.array_equal(nnz_vec(HZp, 1), nnz_vec(HZ, 1)):
        return False, None, None
    if not np.array_equal(nnz_vec(HZp, 0), nnz_vec(HZ, 0)):
        return False, None, None
    return True, HZp, Delta

# ==== Orthogonality check ====
def check_orthogonality(HX: sp.csr_matrix, HZ: sp.csr_matrix, where: str = "") -> bool:
    """
    Prints the number of syndrome violations and returns True iff zero.

    Args:
        HX, HZ: CSR sparse matrices in {0,1}.
        where: Label for logging.

    Returns:
        True if no violations, else False.
    """
    S = compute_syndrome(HX, HZ)
    viol = int(S.nnz)
    print(f"[orth] {where} violations = {viol}")
    return viol == 0

# ==== Iterative driver (sparse) ====
def iterate_random_switches(HX_init: sp.csr_matrix, HZ_init: sp.csr_matrix,
                            n_iterations: int = 5,
                            max_switch_trials: int = 200,
                            enum_limit: int = (1 << 20),
                            add_parity: bool = True):
    """
    Main loop:
      1) Apply one random 2x2 cross-switch to HX.
      2) Build reduced system on K and estimate degrees of freedom.
      3) Repair HZ via ILP to restore orthogonality and preserve weights.
      4) Commit (HX, HZ) and repeat.

    Args:
        HX_init, HZ_init: Initial orthogonal pair (CSR).
        n_iterations: Number of outer iterations.
        max_switch_trials: Max attempts to find a valid 2x2 switch.
        enum_limit: (Reserved) enumeration limit for alternative search.
        add_parity: Whether to include row/col parity constraints in A_big.

    Returns:
        (HX_final, HZ_final) as CSR sparse matrices.
    """
    HX, HZ = HX_init.copy().tocsr(), HZ_init.copy().tocsr()
    assert check_orthogonality(HX, HZ, where="start"), "Initial HX, HZ are not orthogonal."

    for it in range(1, n_iterations + 1):
        print(f"\n===== Iteration {it}/{n_iterations} =====")
        # (A) one random switch on HX
        HXp, move = random_cross_swap_sparse(HX, max_trials=max_switch_trials)
        if move is None:
            print("No valid cross-swap found. Skipping this iteration.")
            continue
        print(f"Applied random switch: {tuple(map(int, move))}")

        I, J, K, _ = extract_I_J_K(HXp, HZ)
        print("I =", I)
        print("J =", J)
        print("K =", K)

        ok_after_swap = check_orthogonality(HXp, HZ, where="after swap")
        if ok_after_swap:
            print("Still orthogonal after swap → No repair needed.")
            HX = HXp
            continue

        # (B) DoF estimate (informative)
        estimate_degrees_of_freedom(HXp, HZ, add_parity=add_parity, verbose=True)

        # (C) Build reduced system
        K_pack = build_system_with_K(HXp, HZ, add_parity=add_parity)

        # (D) ILP repair
        found, HZp, Delta = solve_weight_preserving_ilp(K_pack, HXp, HZ, time_limit_s=10, minimize_flips=True)
        if not found:
            print(f"[Iteration {it}] No feasible HZ' found by ILP — reverting HX.")
            continue

        # (E) Validate
        assert check_orthogonality(HXp, HZp, where="after repair"), "Orthogonality not restored."
        rw_p, rw_0 = nnz_vec(HZp, 1), nnz_vec(HZ, 1)
        cw_p, cw_0 = nnz_vec(HZp, 0), nnz_vec(HZ, 0)
        assert np.array_equal(rw_p, rw_0), "Row weights changed."
        assert np.array_equal(cw_p, cw_0), "Column weights changed."

        # (F) Commit
        HZ, HX = HXp, HZp
        assert check_orthogonality(HX, HZ, where="committed"), "Orthogonality broken after commit."

    assert check_orthogonality(HX, HZ, where="end"), "Final HX, HZ are not orthogonal."
    return HX, HZ

def save_sparse_matrix_png(M: sp.spmatrix, filename: str = "matrix.png", title: str = ""):
    """
    Saves the nonzero pattern of a sparse matrix as a PNG.

    Args:
        M: Sparse matrix to visualize.
        filename: Output PNG path.
        title: Figure title.

    Returns:
        None. Writes a PNG to disk.
    """
    plt.figure(figsize=(8, 8))
    plt.spy(M, markersize=0.5, color="black")
    plt.title(title)
    plt.xlabel("columns")
    plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==== Minimal example ====
if __name__ == "__main__":
    # Example: highly sparse tiled identity
    P, dc, dr = 50, 3, 8
    HX0 = make_block_id_matrix(P, dc, dr)  # (dc*P) × (dr*P)
    HZ0 = HX0.copy()

    print("HX0 shape:", HX0.shape, "nnz:", HX0.nnz)
    print("HZ0 shape:", HZ0.shape, "nnz:", HZ0.nnz)

    HX_final, HZ_final = iterate_random_switches(
        HX0, HZ0,
        n_iterations=1000,
        max_switch_trials=500,
        enum_limit=(1 << 17),
        add_parity=True
    )

    print("\n=== Final ===")
    print("HX_final shape:", HX_final.shape, "nnz:", HX_final.nnz)
    print("HZ_final shape:", HZ_final.shape, "nnz:", HZ_final.nnz)

    # Save PNG snapshots
    save_sparse_matrix_png(HX_final, "HX_final.png", "HX_final structure")
    save_sparse_matrix_png(HZ_final, "HZ_final.png", "HZ_final structure")
    print("Saved HX_final.png and HZ_final.png")
