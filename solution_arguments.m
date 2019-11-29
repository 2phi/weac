function [ args ] = solution_arguments(dat, ci, zi)
%SOLUTION_ARGUMENTS Provide arguments and paths
%   Provide arguments and paths for solution of system of equations by
%   discrimination of kinmatics and boundary conditions

    %% Unwrap input
    
    l = dat.l;
    a = dat.a;

    l1 = dat.l1;
    l2 = dat.l2;
    l3 = dat.l3;

    kn = dat.kn;
    kt = dat.kt;
    mu = dat.mu;
    EI = dat.EI;
    EA = dat.EA;
    lam = dat.lam;
    lam1 = dat.lam1;
    lam2 = dat.lam2;

    qn = dat.qn;
    qt = dat.qt;
    
    bc = dat.bctype;
    kinematics = dat.euler_or_timoshenko;
    
    %% Case discrimination

    if strcmp(bc, 'pst')
        % Propagation saw test boundary condtions
        [c1,c2,c3,c4,c5,c6,c7,c8] = ci{:};
        [z1,z2,z3,z4] = zi{:};
        args.u = {EA,a,kt,l,mu,qt,z1,z2,z3,z4};
        if strcmp(kinematics, 'euler')
            % Euler-Bernoulli beam
            args.w = {EI,a,c1,c2,c3,c4,c5,c6,c7,c8,kn,l,lam,qn};
        elseif strcmp(kinematics, 'timoshenko')
            % Timoshenko beam
            args.w = {EI,a,c1,c2,c3,c4,c5,c6,c7,c8,kn,l,lam1,lam2,qn};
        elseif strcmp(kinematics, 'timoshenko-pasternak')
            % Timoshenko beam on Pasternak foundation
            args.w = {EI,a,c1,c2,c3,c4,c5,c6,c7,c8,kn,l,lam1,lam2,qn};
            args.wp = {EI,a,c1,c2,c3,c4,c6,c7,c8,l,lam1,lam2,qn};
        end
    elseif strcmp(bc, 'slab')
        % Skier triggered crack initiation
        if a == 0
            % Zero crack length
            [c1,c2,c3,c4,c13,c14,c15,c16] = ci{:};
            [z1,z2,z7,z8] = zi{:};
            args.u = {kt,l,l1,mu,qt,z1,z2,z7,z8};
            if strcmp(kinematics, 'euler')
                % Euler-Bernoulli beam
                args.w = {c1,c2,c3,c4,c13,c14,c15,c16,kn,l,l1,lam,qn};
            elseif strcmp(kinematics, 'timoshenko')
                % Timoshenko beam
                args.w = {c1,c2,c3,c4,c13,c14,c15,c16,kn,l,l1,lam1,lam2,qn};
            elseif strcmp(kinematics, 'timoshenko-pasternak')
                % Timoshenko beam on Pasternak foundation
                args.w = {c1,c2,c3,c4,c13,c14,c15,c16,kn,l,l1,lam1,lam2,qn};
                args.wp = {c1,c2,c3,c4,c13,c14,c15,c16,l,l1,lam1,lam2};
            end
        else
            % Nonzero crack length
            [c1,c2,c3,c4,c5,c6,c7,c8,c9,...
                 c10,c11,c12,c13,c14,c15,c16] = ci{:};
            [z1,z2,z3,z4,z5,z6,z7,z8] = zi{:};
            args.u = {EA,kt,l,l1,l2,l3,mu,qt,z1,z2,z3,z4,z5,z6,z7,z8};
            if strcmp(kinematics, 'euler')
                % Euler-Bernoulli beam
                args.w = {EI,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,...
                          c13,c14,c15,c16,kn,l,l1,l2,l3,lam,qn};
            elseif strcmp(kinematics, 'timoshenko')
                % Timoshenko beam
                args.w = {EI,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,...
                          c13,c14,c15,c16,kn,l,l1,l2,l3,lam1,lam2,qn};
            elseif strcmp(kinematics, 'timoshenko-pasternak')
                % Timoshenko beam on Pasternak foundation
                args.w = {EI,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,...
                          c13,c14,c15,c16,kn,l,l1,l2,l3,lam1,lam2,qn};
                args.wp = {EI,c1,c2,c3,c4,c6,c7,c8,c10,c11,c12,...
                           c13,c14,c15,c16,l,l1,l2,l3,lam1,lam2,qn};
            end
        end
    end

end

