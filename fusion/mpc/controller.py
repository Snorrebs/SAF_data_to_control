"""
fusion/mpc/controller.py
Builds and configures the do-mpc NMPC controller for the SAF system.

Objective: minimise (R - r_nom)^2 or (PF - pf_ref)^2 over a horizon H.
Effort:    penalise delta-u to reduce electrode chattering.
Constraint: soft R band [r_nom - r_tol, r_nom + r_tol].
"""
from __future__ import annotations


def build_mpc_controller(
    model,
    H:                  int   = 20,
    pf_ref:             float = 0.75,
    r_nom:              float = 1.0,
    r_tol:              float = 0.07,
    u_min:              float = 0.0,
    u_max:              float = 2.0,
    du_max:             float = 0.01,
    lam_u:              float = 1.0,
    lam_int:            float = 0.0,
    verbose:            bool  = False,
    objective:          str   = "r",
    constraint_penalty: float = 1e4,
    pf_tol:             float = 0.10,
) -> "do_mpc.controller.MPC":
    """Build and return a do-mpc MPC controller for the SAF system."""
    import do_mpc
    import casadi as ca

    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_horizon           = H
    mpc.settings.t_step              = 1.0
    mpc.settings.n_robust            = 0
    mpc.settings.store_full_solution = False

    ipopt_opts = {
        "ipopt.print_level":  0,
        "ipopt.sb":           "yes",
        "print_time":         0,
        "ipopt.max_cpu_time": 15.0,
        "ipopt.max_iter":     1000,
        "ipopt.warm_start_init_point": "yes",
    }
    if not verbose:
        mpc.settings.suppress_ipopt_output = True
    mpc.settings.nlpsol_opts = ipopt_opts

    # mterm cannot contain u (do-mpc restriction), so use state lag-1 R there.
    r1_s = model.aux["r1_pred"]
    r2_s = model.aux["r2_pred"]
    r3_s = model.aux["r3_pred"]
    r1_t = model.x["r1_1"]
    r2_t = model.x["r2_1"]
    r3_t = model.x["r3_1"]

    if objective == "pf" and "pf1_pred" in model.aux.keys():
        pf1   = model.aux["pf1_pred"]
        pf2   = model.aux["pf2_pred"]
        pf3   = model.aux["pf3_pred"]
        lterm = (pf1 - pf_ref)**2 + (pf2 - pf_ref)**2 + (pf3 - pf_ref)**2
        mterm = (r1_t - r_nom)**2 + (r2_t - r_nom)**2 + (r3_t - r_nom)**2
    else:
        lterm = (r1_s - r_nom)**2 + (r2_s - r_nom)**2 + (r3_s - r_nom)**2
        mterm = (r1_t - r_nom)**2 + (r2_t - r_nom)**2 + (r3_t - r_nom)**2

    if lam_int > 0:
        int_cost = lam_int * sum(model.x[f"int{i}"]**2 for i in (1, 2, 3))
        lterm = lterm + int_cost
        mterm = mterm + int_cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u1=lam_u, u2=lam_u, u3=lam_u)

    for ui in ("u1", "u2", "u3"):
        mpc.bounds["lower", "_u", ui] = u_min
        mpc.bounds["upper", "_u", ui] = u_max

    for i in (1, 2, 3):
        r_pred_i = model.aux[f"r{i}_pred"]
        mpc.set_nl_cons(f"r{i}_upper", r_pred_i - r_nom, ub=r_tol,
                        soft_constraint=True, penalty_term_cons=constraint_penalty)
        mpc.set_nl_cons(f"r{i}_lower", r_nom - r_pred_i, ub=r_tol,
                        soft_constraint=True, penalty_term_cons=constraint_penalty)

    # Hard rate constraint: pos{i} = last-step u, so u - pos{i} = dpos this step.
    for i in (1, 2, 3):
        pos_prev = model.x[f"pos{i}"]
        u_i      = model.u[f"u{i}"]
        mpc.set_nl_cons(f"du{i}_up",   u_i - pos_prev, ub=du_max,
                        soft_constraint=False)
        mpc.set_nl_cons(f"du{i}_down", pos_prev - u_i, ub=du_max,
                        soft_constraint=False)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()
    mpc._tvp_template = tvp_template
    return mpc


def update_tvp(
    mpc,
    tca:         float = 0.0,
    tcb:         float = 0.0,
    tcc:         float = 0.0,
    rstd_rx:     "dict | None" = None,
    rstd_r:      "dict | None" = None,
    step_in_win: float = 0.0,
    r_nom:       float = 1.0,
    y_sim:       "dict | None" = None,
) -> None:
    """Update all time-varying parameters before each mpc.make_step() call."""
    tmpl = mpc._tvp_template
    H    = mpc.settings.n_horizon

    rstd_rx = rstd_rx or {1: 0.0, 2: 0.0, 3: 0.0}
    rstd_r  = rstd_r  or {1: 0.0, 2: 0.0, 3: 0.0}
    y_sim   = y_sim   or {1: 1.0, 2: 1.0, 3: 1.0}

    for k in range(H + 1):
        tmpl["_tvp", k, "tca"]         = tca
        tmpl["_tvp", k, "tcb"]         = tcb
        tmpl["_tvp", k, "tcc"]         = tcc
        tmpl["_tvp", k, "step_in_win"] = min(step_in_win + k, H - 1)
        tmpl["_tvp", k, "r_nom"]       = r_nom
        for i in (1, 2, 3):
            tmpl["_tvp", k, f"rstd_rx{i}"] = rstd_rx[i]
            tmpl["_tvp", k, f"rstd_r{i}"]  = rstd_r[i]
            tmpl["_tvp", k, f"y_sim{i}"]   = y_sim[i]
