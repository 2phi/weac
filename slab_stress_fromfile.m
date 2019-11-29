function [ solargs, paths ] = slab_stress_fromfile(dat)
%SLAB_STRESS_FROMFILE Compute deflection, displacement and weak layer stress

    %% Case discriminiation for which system to solve
    
    % Determine which solution to use and compute corresponding input args
    [sysargs, paths, dat.lam1, dat.lam2] = system_arguments_and_paths(dat);
    % Set paths to corresponding matlab functions
    addpath(paths.rod, paths.beam)
    
    
    %% Solve linear systems M*ci = b and C*zi = d for constants
    
    % Beam
    M = beam_coefficientmatrix(sysargs.M{:});    % Coefficient matrix
    n = beam_righthandsidevector(sysargs.n{:});  % Right hand side vector
    ci = num2cell(M\n);                          % Constants vector
    
    % Rod
    C = rod_coefficientmatrix(sysargs.C{:});     % Coefficient matrix
    d = rod_righthandsidevector(sysargs.d{:});   % Right hand side vector
    zi = num2cell(C\d);                          % Constants vector

    
    %% Case discrimination for arguments of u and w

    solargs = solution_arguments(dat, ci, zi);

end

