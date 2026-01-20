import numpy as np
from numba import jit, prange

# ==========================================
# NUMBA KERNELS (MATH & PHYSICS ENGINE)
# ==========================================

@jit(nopython=True, cache=True)
def jit_beam_matrices(xi, yi, xj, yj, E, I, A, phi_s):
    """
    Calculates 2D Frame element stiffness matrix (local & global).
    Supports Timoshenko Shear Deformation via phi_s parameter.
    phi_s = (12 * E * I) / (G * As * L^2) [0.0 for Euler-Bernoulli]
    """
    dx = xj - xi
    dy = yj - yi
    L = np.sqrt(dx**2 + dy**2)
    if L < 1e-6: L = 1e-6
    
    cx = dx / L
    cy = dy / L
    
    # Timoshenko / Euler-Bernoulli Coefficients
    # Common factor includes (1 + phi_s) in denominator for bending terms
    # w_bend_base = E * I / (L^3 * (1 + phi_s))
    
    phi_term = 1.0 + phi_s
    k_bend = (E * I) / (L**3 * phi_term)
    
    w1 = E * A / L             # Axial (EA/L)
    w2 = 12.0 * k_bend         # 12EI / L^3(1+phi)
    w3 = 6.0 * L * k_bend      # 6EI / L^2(1+phi)
    
    # Rotational terms differ in Timoshenko
    # 4EI -> (4 + phi) * EI / L(1+phi) = (4+phi) * L^2 * k_bend
    w4 = (4.0 + phi_s) * L**2 * k_bend
    
    # 2EI -> (2 - phi) * EI / L(1+phi) = (2-phi) * L^2 * k_bend
    w5 = (2.0 - phi_s) * L**2 * k_bend
    
    k_local = np.array([
        [w1, 0, 0, -w1, 0, 0],
        [0, w2, w3, 0, -w2, w3],
        [0, w3, w4, 0, -w3, w5],
        [-w1, 0, 0, w1, 0, 0],
        [0, -w2, -w3, 0, w2, -w3],
        [0, w3, w5, 0, -w3, w4]
    ])
    
    T = np.zeros((6, 6))
    T[0,0] = cx;  T[0,1] = cy
    T[1,0] = -cy; T[1,1] = cx
    T[2,2] = 1.0
    T[3,3] = cx;  T[3,4] = cy
    T[4,3] = -cy; T[4,4] = cx
    T[5,5] = 1.0
    
    k_global = T.T @ k_local @ T
    
    return k_local, k_global, T, L, cx, cy

# --- NON-PRISMATIC LOGIC ---

@jit(nopython=True, cache=True)
def get_I_at_x(x, L, vals, shape_mode, val_mode, b_eff):
    """
    Interpolates Moment of Inertia I at position x.
    vals: array of [v_start, v_mid, v_end] (mid used only if shape_mode=2)
    shape_mode: 0=Constant, 1=Linear(2pt), 2=PiecewiseLinear(3pt)
    val_mode: 0=Values are I, 1=Values are h (I = b_eff/12 * h^3)
    b_eff: Effective width of the section
    """
    v = vals[0]
    xi = x / L
    
    if shape_mode == 0: # Constant
        v = vals[0]
    elif shape_mode == 1: # Linear (Start -> End)
        v = vals[0] * (1 - xi) + vals[1] * xi
    elif shape_mode == 2: # Piecewise Linear (Start -> Mid -> End)
        if xi <= 0.5:
            # Scale 0..0.5 to 0..1
            s = xi * 2.0
            v = vals[0] * (1 - s) + vals[1] * s
        else:
            # Scale 0.5..1.0 to 0..1
            s = (xi - 0.5) * 2.0
            v = vals[1] * (1 - s) + vals[2] * s
            
    if val_mode == 1:
        # v is height, width=b_eff
        return (b_eff * v**3) / 12.0
    else:
        # v is Inertia
        return v

@jit(nopython=True, cache=True)
def jit_non_prismatic_matrices(xi, yi, xj, yj, E, G, shape_mode, val_mode, geom_vals, A_avg, As_avg, b_eff):
    """
    Calculates Stiffness Matrix for non-prismatic beam using Flexibility Method Integration.
    Uses Simpson's Rule (1/3) for exact integration of quadratic flexibility functions.
    """
    dx = xj - xi
    dy = yj - yi
    L = np.sqrt(dx**2 + dy**2)
    if L < 1e-6: L = 1e-6
    cx = dx / L; cy = dy / L
    
    # Numerical Integration Setup (Simpson's Rule - 10 steps / 11 points)
    n_int = 11
    d_step = L / (n_int - 1)
    
    # Flexibility Terms: f11 (rot_i due to M_i), f22 (rot_j due to M_j), f12
    f11 = 0.0; f22 = 0.0; f12 = 0.0
    
    # Shear Flexibility Term (constant approx using As_avg)
    shear_flex_term = 0.0
    if G > 1e-3 and As_avg > 1e-6:
        shear_flex_term = 1.0 / (L * G * As_avg)
    
    for i in range(n_int):
        x = i * d_step
        
        # Simpson's Rule Weights
        if i == 0 or i == n_int - 1:
            weight = 1.0
        elif i % 2 == 1: # Odd
            weight = 4.0
        else: # Even
            weight = 2.0
            
        I_x = get_I_at_x(x, L, geom_vals, shape_mode, val_mode, b_eff)
        EI = E * I_x
        
        # Unit Moment Diagrams: m1 = (1 - x/L), m2 = (x/L)
        m1 = 1.0 - x/L
        m2 = x/L
        
        # Simpson's Integration Factor
        fac = (weight * d_step / 3.0) / EI
        
        f11 += m1 * m1 * fac
        f22 += m2 * m2 * fac
        f12 += m1 * m2 * fac
        
    # Add Shear Flexibility
    f11 += shear_flex_term
    f22 += shear_flex_term
    f12 -= shear_flex_term 
    
    # Invert F to get K_rot
    det = f11 * f22 - f12 * f12
    if abs(det) < 1e-12: det = 1e-12
    
    k11 = f22 / det
    k22 = f11 / det
    k12 = -f12 / det
    
    # Build 6x6 Local Stiffness
    w_A = E * A_avg / L
    
    # Shear forces from moment equilibrium: V = (Mi + Mj)/L
    q1 = (k11 + k12) / L
    q2 = (k12 + k22) / L
    q3 = (q1 + q2) / L
    
    k_loc = np.zeros((6,6))
    
    # Axial
    k_loc[0,0] = w_A;  k_loc[0,3] = -w_A
    k_loc[3,0] = -w_A; k_loc[3,3] = w_A
    
    # Bending / Shear
    k_loc[1,1] = q3;   k_loc[1,2] = q1;   k_loc[1,4] = -q3;  k_loc[1,5] = q2
    k_loc[2,1] = q1;   k_loc[2,2] = k11;  k_loc[2,4] = -q1;  k_loc[2,5] = k12
    k_loc[4,1] = -q3;  k_loc[4,2] = -q1;  k_loc[4,4] = q3;   k_loc[4,5] = -q2
    k_loc[5,1] = q2;   k_loc[5,2] = k12;  k_loc[5,4] = -q2;  k_loc[5,5] = k22
    
    # Transform
    T = np.zeros((6, 6))
    T[0,0] = cx;  T[0,1] = cy
    T[1,0] = -cy; T[1,1] = cx
    T[2,2] = 1.0
    T[3,3] = cx;  T[3,4] = cy
    T[4,3] = -cy; T[4,4] = cx
    T[5,5] = 1.0
    
    k_glob = T.T @ k_loc @ T
    
    return k_loc, k_glob, T, L, cx, cy

# --- NEW: NUMERICAL LOAD INTEGRATION (CONSISTENT NODAL LOADS) ---

@jit(nopython=True, cache=True)
def get_static_M0_V0_point(x, L, P, a):
    """Static Moment/Shear for Simply Supported beam with Point Load."""
    if x < a:
        # Reaction at start: R_i = P*(L-a)/L
        R_i = P * (1.0 - a/L)
        V0 = R_i
        M0 = R_i * x
    else:
        # Reaction at start same
        R_i = P * (1.0 - a/L)
        V0 = R_i - P
        M0 = R_i * x - P * (x - a)
    return M0, V0

@jit(nopython=True, cache=True)
def get_static_M0_V0_trapezoid(x, L, q_s, q_e, h_s, h_e):
    """Static Moment/Shear for Simply Supported beam with Trapezoid Load."""
    
    # 1. Calculate Total Reactions R_i, R_j for simply supported beam
    # Load Resultant F
    len_load = h_e - h_s
    if len_load < 1e-9: return 0.0, 0.0
    
    F_res = (q_s + q_e) / 2.0 * len_load
    
    # Centroid from h_s
    # x_c_local = (L/3) * (2*q_e + q_s) / (q_e + q_s)
    denom = q_e + q_s
    if abs(denom) < 1e-9:
        x_c_local = len_load / 2.0
    else:
        x_c_local = (len_load / 3.0) * (2.0 * q_e + q_s) / denom
        
    x_c_global = h_s + x_c_local
    
    # Reactions
    R_j = F_res * x_c_global / L
    R_i = F_res - R_j
    
    # 2. Calculate Internal Forces at x
    if x <= h_s:
        V0 = R_i
        M0 = R_i * x
    elif x >= h_e:
        V0 = R_i - F_res
        M0 = R_i * x - F_res * (x - x_c_global)
    else:
        # Inside the load
        L_prime = x - h_s
        # q at x
        q_x = q_s + (q_e - q_s) * (L_prime / len_load)
        
        # Resultant of part up to x
        F_prime = (q_s + q_x) / 2.0 * L_prime
        
        # Centroid of F_prime (distance from x back to centroid)
        denom_p = q_x + q_s
        if abs(denom_p) < 1e-9:
            dist_back = L_prime / 2.0
        else:
            dist_back = (L_prime / 3.0) * (q_x + 2.0*q_s) / denom_p
            
        V0 = R_i - F_prime
        M0 = R_i * x - F_prime * dist_back
        
    return M0, V0

@jit(nopython=True, cache=True)
def jit_numerical_fef(load_type, params, L, E, G, shape_mode, val_mode, geom_vals, b_eff, As_avg):
    """
    Calculates Fixed End Forces (FEF) using Numerical Integration (Simpson's Rule).
    Must match jit_non_prismatic_matrices integration exactly.
    load_type: 1=Point, 2=Trapezoid
    params: [P, a] or [q_s, q_e, h_s, h_e]
    """
    # Simpson's Setup
    n_int = 11
    d_step = L / (n_int - 1)
    
    f11 = 0.0; f22 = 0.0; f12 = 0.0
    theta_i = 0.0; theta_j = 0.0
    
    shear_flex_term = 0.0
    has_shear = (G > 1e-3 and As_avg > 1e-6)
    if has_shear:
        shear_flex_term = 1.0 / (L * G * As_avg)
    
    # Integration Loop
    for i in range(n_int):
        x = i * d_step
        
        # Weights
        if i == 0 or i == n_int - 1: weight = 1.0
        elif i % 2 == 1: weight = 4.0
        else: weight = 2.0
            
        I_x = get_I_at_x(x, L, geom_vals, shape_mode, val_mode, b_eff)
        EI = E * I_x
        
        fac = (weight * d_step / 3.0) / EI
        
        m1 = 1.0 - x/L
        m2 = x/L
        
        # Flexibility Accumulation (Same as stiffness kernel)
        f11 += m1 * m1 * fac
        f22 += m2 * m2 * fac
        f12 += m1 * m2 * fac
        
        # Load Vector Accumulation
        # Get M0(x) and V0(x) (Simply Supported Forces)
        if load_type == 1: # Point
            M0, V0 = get_static_M0_V0_point(x, L, params[0], params[1])
        elif load_type == 2: # Trapezoid
            M0, V0 = get_static_M0_V0_trapezoid(x, L, params[0], params[1], params[2], params[3])
        else:
            M0, V0 = 0.0, 0.0
            
        # Work Done: Integral (M0 * m / EI)
        theta_i += M0 * m1 * fac
        theta_j += M0 * m2 * fac
        
        # Shear Work for Load Vector: Integral (V0 * v / GAs)
        # v1 = -1/L, v2 = 1/L
        if has_shear:
            # Shear work uses simplified constant area integration for consistency with stiffness kernel
            # Factor: (weight * d_step / 3.0) / (G * As_avg)
            fac_s = (weight * d_step / 3.0) / (G * As_avg)
            theta_i += V0 * (-1.0/L) * fac_s
            theta_j += V0 * (1.0/L) * fac_s

    # Add Shear Flexibility to Matrix
    f11 += shear_flex_term
    f22 += shear_flex_term
    f12 -= shear_flex_term
    
    # Solve System F * M = -Theta
    # | f11 f12 | | Mi | = | -theta_i |
    # | f12 f22 | | Mj |   | -theta_j |
    
    det = f11 * f22 - f12 * f12
    if abs(det) < 1e-12: det = 1e-12
    
    # Inverse of 2x2 F
    k11 = f22 / det
    k22 = f11 / det
    k12 = -f12 / det
    
    # FEF Moments
    Mi = -(k11 * theta_i + k12 * theta_j)
    Mj = -(k12 * theta_i + k22 * theta_j)
    
    # FEF Shears (Equilibrium)
    # Total Reaction = Simple Reaction + (Mi + Mj)/L
    # But we need to calculate Simple Reaction total first?
    # Actually, we can get total load F_total and verify. 
    # Or just use the M0/V0 at ends.
    # R_i_fixed = R_i_simple + (Mi + Mj)/L
    # R_j_fixed = R_j_simple - (Mi + Mj)/L
    
    # Recalculate simple reactions R_i0, R_j0 for the whole beam
    if load_type == 1: # Point
        P, a = params[0], params[1]
        Ri0 = P * (1.0 - a/L)
        Rj0 = P * a/L
    elif load_type == 2: # Trapezoid
        q_s, q_e, h_s, h_e = params[0], params[1], params[2], params[3]
        len_load = h_e - h_s
        F_res = (q_s + q_e) / 2.0 * len_load
        if abs(q_e + q_s) < 1e-9: xc = len_load/2.0
        else: xc = (len_load/3.0)*(2*q_e + q_s)/(q_e + q_s)
        xg = h_s + xc
        Rj0 = F_res * xg / L
        Ri0 = F_res - Rj0
    else:
        Ri0, Rj0 = 0.0, 0.0
        
    Vi = Ri0 + (Mi + Mj) / L
    Vj = Rj0 - (Mi + Mj) / L
    
    # Return 6-DOF vector [Ni, Vi, Mi, Nj, Vj, Mj]
    # Note: FEF vector is forces EXERTED BY NODES ON BEAM? 
    # Usually FEF is "Fixed End Actions" = forces exerted BY BEAM ON NODES.
    # Stiffness eqn: K*u = F_external + F_equivalent_nodal
    # F_equivalent = - FEF.
    # The existing jit_fef functions return "Reaction-like" forces (Forces on Nodes).
    # e.g. jit_fef_point returns positive R1 upwards.
    # So we return Vi, Mi, Vj, Mj as defined.
    
    f_vec = np.zeros(6)
    f_vec[1] = Vi
    f_vec[2] = Mi
    f_vec[4] = Vj
    f_vec[5] = Mj
    
    return f_vec

# --- STANDARD FEF (ANALYTICAL) ---

@jit(nopython=True, cache=True)
def jit_fef_point(P, a, L):
    f = np.zeros(6)
    b = L - a
    if 0 <= a <= L:
        m1 = P * a * b**2 / L**2
        m2 = -P * a**2 * b / L**2
        R1 = P * b**2 * (3*a + b) / L**3
        R2 = P * a**2 * (a + 3*b) / L**3
        f[1] = R1
        f[2] = m1
        f[4] = R2
        f[5] = m2
    return f

@jit(nopython=True, cache=True)
def jit_fef_trapezoid(q_s, q_e, h_s, h_e, L):
    f = np.zeros(6)
    if h_e > L: h_e = L
    if h_s < 0: h_s = 0
    if h_s >= h_e: return f

    num_int = 20
    step = (h_e - h_s) / (num_int - 1)
    
    F_equiv = np.zeros(4) 
    
    for i in range(num_int):
        x = h_s + i * step
        weight = 1.0 if (i == 0 or i == num_int - 1) else 2.0
        wx = q_s + (q_e - q_s) * (x - h_s) / (h_e - h_s)
        
        xi = x/L
        xi2 = xi*xi
        xi3 = xi2*xi
        
        n_v1 = 1.0 - 3.0*xi2 + 2.0*xi3
        n_m1 = x * (1.0 - xi)**2
        n_v2 = 3.0*xi2 - 2.0*xi3
        n_m2 = x * (xi2 - xi)
        
        val = weight * wx * step / 2.0
        
        F_equiv[0] += val * n_v1
        F_equiv[1] += val * n_m1
        F_equiv[2] += val * n_v2
        F_equiv[3] += val * n_m2
        
    f[1] = F_equiv[0]
    f[2] = F_equiv[1]
    f[4] = F_equiv[2]
    f[5] = F_equiv[3]
    return f

@jit(nopython=True, cache=True)
def jit_internal_forces(L, f_start, num_pts, load_data):
    x_vals = np.linspace(0, L, num_pts)
    M_vals = np.zeros(num_pts)
    V_vals = np.zeros(num_pts)
    N_vals = np.zeros(num_pts)
    
    N0, V0, M0 = f_start[0], f_start[1], f_start[2]
    
    for i in range(num_pts):
        x = x_vals[i]
        Mx = M0 - V0 * x
        Vx = V0
        Nx = -N0 
        
        for j in range(len(load_data)):
            l_type = int(load_data[j, 0])
            
            if l_type == 2:
                q_s, q_e, h_s, h_e = load_data[j, 1], load_data[j, 2], load_data[j, 3], load_data[j, 4]
                if x > h_s:
                    lim = x if x < h_e else h_e
                    len_eff = lim - h_s
                    
                    if abs(h_e - h_s) > 1e-9:
                        q_at_lim = q_s + (q_e - q_s) * (len_eff) / (h_e - h_s)
                        F_part = (q_s + q_at_lim) / 2.0 * len_eff
                        Vx -= F_part
                        
                        denom = q_s + q_at_lim
                        if abs(denom) > 1e-9:
                            d_c = (len_eff / 3.0) * (2.0 * q_s + q_at_lim) / denom
                        else:
                            d_c = 0.0
                        dist_arm = (x - h_s) - d_c
                        Mx += F_part * dist_arm

            elif l_type == 1:
                P, a = load_data[j, 1], load_data[j, 2]
                if x > a:
                    Vx -= P
                    Mx += P * (x - a)
                    
        M_vals[i] = Mx
        V_vals[i] = Vx
        N_vals[i] = Nx
        
    return x_vals, M_vals, V_vals, N_vals

@jit(nopython=True, cache=True)
def jit_disp_shape(u_vec, L, num_pts):
    x = np.linspace(0, L, num_pts)
    xi = x / L
    u_x = u_vec[0] * (1 - xi) + u_vec[3] * xi
    phi1 = 1 - 3*xi**2 + 2*xi**3
    phi2 = x * (1 - xi)**2
    phi3 = 3*xi**2 - 2*xi**3
    phi4 = x * (xi**2 - xi)
    u_y = phi1*u_vec[1] + phi2*u_vec[2] + phi3*u_vec[4] + phi4*u_vec[5]
    return u_x, u_y

@jit(nopython=True, cache=True)
def jit_annotation_solver(data_arr):
    N = len(data_arr)
    iterations = 30
    for _ in range(iterations):
        moves_x = np.zeros(N)
        moves_y = np.zeros(N)
        collision = False
        for i in range(N):
            for j in range(i + 1, N):
                ax, ay, aw, ah = data_arr[i, 0], data_arr[i, 1], data_arr[i, 2], data_arr[i, 3]
                bx, by, bw, bh = data_arr[j, 0], data_arr[j, 1], data_arr[j, 2], data_arr[j, 3]
                
                dx = abs(ax - bx)
                dy = abs(ay - by)
                min_dx = (aw + bw) * 0.75
                min_dy = (ah + bh) * 0.75
                
                if dx < min_dx and dy < min_dy:
                    collision = True
                    overlap_x = min_dx - dx
                    overlap_y = min_dy - dy
                    a_perp_x, a_perp_y = data_arr[i, 4], data_arr[i, 5]
                    is_vert_a = abs(a_perp_y) < abs(a_perp_x)
                    
                    if is_vert_a:
                        direction = 1.0 if ax > bx else -1.0
                        moves_x[i] += overlap_x * 0.5 * direction
                        moves_x[j] -= overlap_x * 0.5 * direction
                    else:
                        direction = 1.0 if ay > by else -1.0
                        moves_y[i] += overlap_y * 0.5 * direction
                        moves_y[j] -= overlap_y * 0.5 * direction
        if not collision: break
        for k in range(N):
            data_arr[k, 0] += moves_x[k]
            data_arr[k, 1] += moves_y[k]
    return data_arr

# ----------------------------------------
# OPTIMIZED BATCH KERNELS
# ----------------------------------------

@jit(nopython=True, cache=True)
def jit_build_batch_F(NDOF, n_steps, x_steps, v_loads, v_dists, sp_start_x, sp_lens, sp_el_indices, el_L, el_T, el_dof_indices):
    F_all = np.zeros((NDOF, n_steps))
    num_axles = len(v_loads)
    num_spans = len(sp_start_x)
    
    for i in range(n_steps):
        x_front = x_steps[i]
        for j in range(num_axles):
            load_x = x_front - v_dists[j]
            for k in range(num_spans):
                s_x = sp_start_x[k]
                e_x = s_x + sp_lens[k]
                if s_x <= load_x <= e_x:
                    local_x = load_x - s_x
                    el_idx = sp_el_indices[k]
                    # NOTE: Moving load stepping typically uses Point Load kernel.
                    # For consistency in tapered elements, this should also ideally use the numerical kernel.
                    # However, batch stepping performance is critical. 
                    # For now, we retain analytical for speed, accepting minor inconsistency in moving loads
                    # unless specified otherwise.
                    f_local = jit_fef_point(v_loads[j], local_x, el_L[el_idx])
                    T = el_T[el_idx]
                    f_global_vec = T.T @ f_local
                    dofs = el_dof_indices[el_idx]
                    for d in range(6):
                        F_all[dofs[d], i] -= f_global_vec[d]
                    break
    return F_all

@jit(nopython=True, cache=True)
def jit_precompute_stress_recovery(n_elems, n_pts, el_L, el_k_local):
    S = np.zeros((n_elems, n_pts, 5, 6))
    for e in range(n_elems):
        L = el_L[e]
        k = el_k_local[e]
        dx = L / (n_pts - 1)
        for p in range(n_pts):
            x = p * dx
            xi = x / L
            for j in range(6):
                S[e, p, 0, j] = k[2, j] - x * k[1, j] # M
                S[e, p, 1, j] = k[1, j]               # V
                S[e, p, 2, j] = -k[0, j]              # N
            S[e, p, 3, 0] = 1.0 - xi
            S[e, p, 3, 3] = xi
            xi2 = xi*xi
            xi3 = xi2*xi
            phi1 = 1.0 - 3.0*xi2 + 2.0*xi3
            phi2 = x * (1.0 - xi)**2
            phi3 = 3.0*xi2 - 2.0*xi3
            phi4 = x * (xi2 - xi)
            S[e, p, 4, 1] = phi1
            S[e, p, 4, 2] = phi2
            S[e, p, 4, 4] = phi3
            S[e, p, 4, 5] = phi4
    return S

@jit(nopython=True, parallel=True, cache=True)
def jit_envelope_batch_parallel(
    n_steps, n_elems, n_pts, 
    x_steps, v_loads, v_dists, 
    sp_start_x, sp_lens, sp_el_indices, 
    D_all, el_dof_indices, el_T, el_L, 
    S_matrices,
    res_accum,
    is_init
):
    num_axles = len(v_loads)
    num_spans = len(sp_start_x)
    
    for e in prange(n_elems):
        if is_init:
            for p in range(n_pts):
                res_accum[e, p, 0] = -1e15; res_accum[e, p, 1] = 1e15 # M
                res_accum[e, p, 2] = -1e15; res_accum[e, p, 3] = 1e15 # V
                res_accum[e, p, 4] = -1e15; res_accum[e, p, 5] = 1e15 # N
                res_accum[e, p, 6] = -1e15; res_accum[e, p, 7] = 1e15 # dx
                res_accum[e, p, 8] = -1e15; res_accum[e, p, 9] = 1e15 # dy

        L_el = el_L[e]
        T = el_T[e]
        cx, cy = T[0,0], T[0,1]
        dofs = el_dof_indices[e]
        
        span_idx = -1
        span_offset = 0.0
        for sp in range(num_spans):
            if sp_el_indices[sp] == e:
                span_idx = sp
                span_offset = sp_start_x[sp]
                break
        
        s_start = 0.0
        s_end = 0.0
        has_span = False
        if span_idx >= 0:
            has_span = True
            s_start = span_offset
            s_end = s_start + sp_lens[span_idx]

        dx_step = L_el / (n_pts - 1)
        
        d_glob = np.zeros(6)
        d_loc = np.zeros(6)
        fef_total = np.zeros(6)

        for s in range(n_steps):
            x_front = x_steps[s]
            for k in range(6): d_glob[k] = D_all[dofs[k], s]
            
            for r in range(6):
                val = 0.0
                for c in range(6): val += T[r, c] * d_glob[c]
                d_loc[r] = val
            
            fef_total[:] = 0.0
            if has_span:
                for ax in range(num_axles):
                    load_x_glob = x_front - v_dists[ax]
                    if s_start <= load_x_glob <= s_end:
                        local_x = load_x_glob - s_start
                        fef_total += jit_fef_point(v_loads[ax], local_x, L_el)
            
            FEF_N, FEF_V, FEF_M = fef_total[0], fef_total[1], fef_total[2]
            
            for p in range(n_pts):
                x = p * dx_step
                M_hom = 0.0; V_hom = 0.0; N_hom = 0.0; ux_loc = 0.0; uy_loc = 0.0
                
                for j in range(6):
                    dj = d_loc[j]
                    M_hom += S_matrices[e, p, 0, j] * dj
                    V_hom += S_matrices[e, p, 1, j] * dj
                    N_hom += S_matrices[e, p, 2, j] * dj
                    ux_loc += S_matrices[e, p, 3, j] * dj
                    uy_loc += S_matrices[e, p, 4, j] * dj

                Mx = M_hom + (FEF_M - FEF_V * x)
                Vx = V_hom + FEF_V
                Nx = N_hom - FEF_N 
                
                if has_span:
                    for ax in range(num_axles):
                        load_x_glob = x_front - v_dists[ax]
                        if s_start <= load_x_glob <= s_end:
                            a = load_x_glob - s_start
                            if x > a:
                                P = v_loads[ax]
                                Vx -= P
                                Mx += P * (x - a)
                
                def_x_glob = cx * ux_loc - cy * uy_loc
                def_y_glob = cy * ux_loc + cx * uy_loc
                
                if Mx > res_accum[e, p, 0]: res_accum[e, p, 0] = Mx
                if Mx < res_accum[e, p, 1]: res_accum[e, p, 1] = Mx
                if Vx > res_accum[e, p, 2]: res_accum[e, p, 2] = Vx
                if Vx < res_accum[e, p, 3]: res_accum[e, p, 3] = Vx
                if Nx > res_accum[e, p, 4]: res_accum[e, p, 4] = Nx
                if Nx < res_accum[e, p, 5]: res_accum[e, p, 5] = Nx
                if def_x_glob > res_accum[e, p, 6]: res_accum[e, p, 6] = def_x_glob
                if def_x_glob < res_accum[e, p, 7]: res_accum[e, p, 7] = def_x_glob
                if def_y_glob > res_accum[e, p, 8]: res_accum[e, p, 8] = def_y_glob
                if def_y_glob < res_accum[e, p, 9]: res_accum[e, p, 9] = def_y_glob