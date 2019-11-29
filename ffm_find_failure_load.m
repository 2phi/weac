function [m_f, da_f] = ffm_find_failure_load(m0_max, data, s_crit, G_crit)
    %Iteration:
    tol = 1e-2;             % Convergence criterion
    d = 0.2;                % Load update damping 0 <= d < 1
    maxiter = 50;           % Maximum iterations
    options = optimset('TolX', 2.2204e-16);
    
    % Initialize timer
    globaltimer = tic;

    % Initialize load, crack length and ERR arrays
    [m, da, Gbari, Gbarii, x0, t] = deal(nan(maxiter, 1));
    fprintf_verbosity('-------------------------------------\n')
    fprintf_verbosity('Iter   m [kg]     da [mm]   time [ms]\n')
    fprintf_verbosity('-------------------------------------\n')

    % Compute and interpolate stress solution for unit load
    unitload = ffm_init(data);

    % Compute a0 such that stress criterion is satisfied for m0
    for m0 = [1 5:5:195 linspace(200, m0_max, 10)]
        % Stresses for load m0 in kg
        sig = @(x) m0*ppval(unitload.sig, x) - data.qn/data.b;
        tau = @(x) m0*ppval(unitload.tau, x) + data.qt/data.b;
        fstress = crit_func_s(data, s_crit, sig, tau);
        try
            x0(1) = fzero(fstress, [0 data.l/2]);
            break
        catch
%             fprintf('    m0 = %.0f too low, trying higher\n', m0)
        end
    end
    m(1) = m0;
    da(1) = 2*abs(data.l/2 - x0(1));  % works for x0 on either side of peak

    % Print first iteration
    t(1) = 1e3*toc(globaltimer);
    fprintf_verbosity('%2i    %7.2f   %8.2f      %5.1f\n', ...
            1, m(1), da(1), t(1))

    % Compute IERR for intial crack da(1) and initial loading m(1)
    [Gbari(1), Gbarii(1)] = ffm_ierr(m(1), da(1), unitload, data);

    % Iterative solver
    for n = 2:maxiter
        timer = tic;
        % Update load
        g_crit_val = crit_func_G(Gbari(n-1), Gbarii(n-1), G_crit, data);
        update = 1/g_crit_val;
        m(n) = (update^0.5)^(1-d)*m(n-1);
        % Update stress criterion with new load
        sig = @(x) m(n)*ppval(unitload.sig, x) - data.qn/data.b;
        tau = @(x) m(n)*ppval(unitload.tau, x) + data.qt/data.b;
        fstress = crit_func_s(data, s_crit, sig, tau);
        % Update crack length
        try
            x0(n) = fzero(fstress, x0(n-1), options);
        catch
            fprintf('  XXX No root found for fstress\n')
            m_f = NaN;
            da_f = NaN;
            break
        end
        da(n) = 2*abs(data.l/2 - x0(n));
        
        % Print iteration
        t(n) = 1e3*toc(globaltimer);
        fprintf_verbosity('%2i    %7.2f   %8.2f      %5.1f\n', ...
                  n, m(n), da(n), 1e3*toc(timer))
        % Check convergence
        if abs(m(n) - m(n-1))/m(n) < tol
            m_f = m(n);
            da_f = da(n);
            break
        end
        if n == maxiter
            fprintf('  XXX No convergence found within maxiter \n')
            fprintf(' \n')
            m_f = NaN;
            da_f = NaN;
            break
        end
        [Gbari(n), Gbarii(n)] = ffm_ierr(m(n), da(n), unitload, data);
    end

    % Critical load
    F_f = m_f*1e-3*data.g;    % Load [N]
    
    %% Final values
    time_f = 1e3*toc(globaltimer);

    % Results output
    fprintf_verbosity('------------------------------------\n')
    fprintf_verbosity('Critical skier weight:   %0.2f kg\n', m_f)
    fprintf_verbosity('Critical load:           %0.2f N\n', F_f)
    fprintf_verbosity('Crack length:            %0.2f mm\n', da_f) 
    fprintf_verbosity('Total time:              %0.1f ms\n', time_f)
    fprintf_verbosity('\n')

end
