function regularize_Q!(solver::RelaxationSolver{T}) where {T}
    if solver.opts.q_regularization == :none || solver.ipm.mu < solver.opts.min_reg_mu
        return false
    end

    ipm = solver.ipm
    cb = ipm.cb
    rnlp = solver.rnlp
    kkt = solver.ipm.kkt
    n = length(ipm.x_ur)
    ncc = get_ncc(solver.mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    A = Array{T}(undef, 2, 2)
    regularized = false
    for i in 1:ncc
        cc1 = ind_cc1[i]
        cc2 = ind_cc2[i]
        ys = ipm.y[end-ncc+i]
        scale = cb.con_scale[end-ncc+i]

        if solver.opts.q_regularization == :eigenvalue_decomposition
            A[1, 1] = kkt.pr_diag[cc1]
            A[2, 2] = kkt.pr_diag[cc2]
            A[2, 1] = ys*scale
            A[1, 2] = ys*scale
            E = eigen(Symmetric(A))
            if E.values[1] < 0 || E.values[2] > solver.opts.max_eig_value
                E.values[1] = solver.opts.min_eig_value
                E.values[2] = min(solver.opts.max_eig_value, E.values[2])
                A .= Symmetric(Matrix(E))
                kkt.reg[cc1] = A[1, 1] - kkt.pr_diag[cc1]
                kkt.reg[cc2] = A[2, 2] - kkt.pr_diag[cc2]
                kkt.pr_diag[cc1] = A[1, 1]
                kkt.pr_diag[cc2] = A[2, 2]
                kkt.hess_raw.V[end-ncc+i] = A[1, 2]
                regularized = true
            end
        elseif solver.opts.q_regularization == :critical_rho
            ys_max = sqrt(kkt.pr_diag[cc1]*kkt.pr_diag[cc2])

            if ys*scale > ys_max
                kkt.hess_raw.V[end-ncc+i] =
                    solver.opts.critical_rho_factor*ys_max*(
                        rnlp.meta.minimize ? one(T) : -one(T)
                    )
                regularized = true
            end
        end
    end
    # We modify hess_raw so need to compress_hessian again.
    MadNLPCore.compress_hessian!(kkt)
    return regularized
end

function unregularize_Q!(solver::RelaxationSolver{T}) where {T}
    ipm = solver.ipm
    cb = ipm.cb
    rnlp = solver.rnlp
    kkt = solver.ipm.kkt
    n = length(ipm.x_ur)
    ncc = get_ncc(solver.mpcc)
    nnzh = get_nnzh(solver.mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    A = Array{T}(undef, 2, 2)
    regularized = false
    kkt.pr_diag[ind_cc1] .-= kkt.reg[ind_cc1]
    kkt.pr_diag[ind_cc2] .-= kkt.reg[ind_cc2]
    kkt.reg[ind_cc1] .= 0
    kkt.reg[ind_cc2] .= 0
    kkt.hess_raw.V[(end-ncc+1):end] .=
        ipm.y[(end-ncc+1):end] .* cb.con_scale[(end-ncc+1):end] # TODO(@anton): this may allocate
    # We modify hess_raw so need to compress_hessian again.
    MadNLPCore.compress_hessian!(kkt)
    return regularized
end

function MadNLPCore.inertia_correction!(
    inertia_corrector::MadNLPCore.InertiaBased,
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    n_trial = 0
    ipm.del_w = del_w_prev = zero(T)
    ipm.del_c = del_c_prev = zero(T)

    MadNLPCore.@trace(ipm.logger, "Inertia-based regularization started.")

    MadNLPCore.factorize_wrapper!(ipm)
    num_pos, num_zero, num_neg = MadNLPCore.inertia(ipm.kkt.linear_solver)

    solve_status = if MadNLPCore.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
        # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
        MadNLPCore.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
    else
        false
    end

    # Try to regularize
    if !solve_status && regularize_Q!(solver)
        MadNLPCore.@trace(ipm.logger, "Trying to regularize out penalty.")
        MadNLPCore.factorize_wrapper!(ipm)
        num_pos, num_zero, num_neg = MadNLPCore.inertia(ipm.kkt.linear_solver)

        solve_status = if MadNLPCore.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
            # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
            MadNLPCore.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
        else
            false
        end

        if !solve_status
            # undo what we did and just do inertia correction
            unregularize_Q!(solver)
        end
    end
    while !solve_status
        MadNLPCore.@debug(ipm.logger, "Primal-dual perturbed.")

        if n_trial == 0
            ipm.del_w =
                ipm.del_w_last==zero(T) ? ipm.opt.first_hessian_perturbation :
                max(
                    ipm.opt.min_hessian_perturbation,
                    ipm.opt.perturb_dec_fact*ipm.del_w_last,
                )
        else
            ipm.del_w *=
                ipm.del_w_last==zero(T) ? ipm.opt.perturb_inc_fact_first :
                ipm.opt.perturb_inc_fact
            if ipm.del_w>ipm.opt.max_hessian_perturbation
                ipm.cnt.k+=1
                MadNLPCore.@debug(
                    ipm.logger,
                    "Primal regularization is too big. Switching to restoration phase."
                )
                return false
            end
        end
        ipm.del_c =
            num_zero == 0 ? zero(T) :
            ipm.opt.jacobian_regularization_value *
            ipm.mu^(ipm.opt.jacobian_regularization_exponent)
        MadNLPCore.regularize_diagonal!(ipm.kkt, ipm.del_w - del_w_prev, ipm.del_c - del_c_prev)
        del_w_prev = ipm.del_w
        del_c_prev = ipm.del_c

        MadNLPCore.factorize_wrapper!(ipm)
        num_pos, num_zero, num_neg = MadNLPCore.inertia(ipm.kkt.linear_solver)

        solve_status = if MadNLPCore.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
            MadNLPCore.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
        else
            false
        end

        n_trial += 1
    end

    ipm.del_w != 0 && (ipm.del_w_last = ipm.del_w)
    return true
end

function MadNLPCore.inertia_correction!(
    inertia_corrector::MadNLPCore.AbstractInertiaCorrector,
    solver::RelaxationSolver{T},
) where {T}
    return MadNLPCore.inertia_correction!(inertia_corrector, solver.ipm)
end
