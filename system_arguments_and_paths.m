function [ args, paths, lam1, lam2 ] = system_arguments_and_paths(dat)
%SYSTEM_ARGUMENTS_AND_PATHS Provide arguments and paths
%   Provide arguments and paths for solution of system of equations by
%   discrimination of kinematics, boundary conditions and material
%   paramters

    %% Unwrap input
    
    l = dat.l;
    a = dat.a;

    l1 = dat.l1;
    l2 = dat.l2;
    l3 = dat.l3;
    l4 = dat.l4;

    kn = dat.kn;
    kt = dat.kt;
    ks = dat.ks;
    mu = dat.mu;
    EI = dat.EI;
    EA = dat.EA;
    kGA = dat.kGA;
    lam = dat.lam;

    qn = dat.qn;
    qt = dat.qt;
    Fn = dat.Fn;
    Ft = dat.Ft;
    Mr = dat.Mr;
    Vr = dat.Vr;
    Nr = dat.Nr;
    
    bc = dat.bctype;
    kinematics = dat.euler_or_timoshenko;

    
    %%  Rod case discrimination
    
    if strcmp(bc, 'pst')
        % Propagation saw test boundary condtions
        paths.rod = 'symbolic/rod/pst';
        args.C = {a,l,mu};
%         args.d = {EA,a,kt,qt};
        args.d = {EA,Nr,a,kt,qt};
    elseif strcmp(bc, 'slab')
        % Skier triggered crack initiation boundary conditions
        if a == 0
            % Zero crack length
            paths.rod = 'symbolic/rod/nocrack';
            args.C = {l1,l4,mu};
            args.d = {EA,Ft};
        else
            % Nonzero crack length
            paths.rod = 'symbolic/rod/crack';
            args.C = {l1,l2,l3,l4,mu};
            args.d = {EA,Ft,kt,l2,l3,qt};
        end
    end


    %% Beam case discrimination
    
    % Euler-Bernoulli beam
    if strcmp(kinematics, 'euler')
        lam1 = 0;
        lam2 = 0;
        if strcmp(bc, 'pst')
            % Propagation saw test boundary condtions
            paths.beam = 'symbolic/beam/euler/pst';
            args.M = {a, l, lam};
            args.n = {EI,Mr,Vr,a,kn,qn};
        elseif strcmp(bc, 'slab')
            % Skier triggered crack initiation boundary conditions
            if a == 0
                % Zero crack length
                paths.beam = 'symbolic/beam/euler/nocrack';
                args.M = {l1,l4,lam};
                args.n = {EI,Fn};
            else
                % Nonzero crack length
                paths.beam = 'symbolic/beam/euler/crack';
                args.M = {l1,l2,l3,l4,lam};
                args.n = {EI,Fn,kn,l2,l3,qn};
            end
        end
    % Timoshenko beam
    elseif strcmp(kinematics, 'timoshenko')
        if kn*EI < 4*kGA^2
            % Roots of characteristic polynomial are complex
            lam1 = sqrt(sqrt(kn/(4*EI)) + kn/(4*kGA));
            lam2 = sqrt(sqrt(kn/(4*EI)) - kn/(4*kGA));
            if strcmp(bc, 'pst')
                % Propagation saw test boundary condtions
                paths.beam = 'symbolic/beam/timoshenko/complex/pst';
                args.M = {EI,a,kGA,kn,l,lam1,lam2};
%                 args.n = {EI,a,kGA,kn,qn};
                args.n = {EI,Mr,Vr,a,kGA,kn,qn};
            elseif strcmp(bc, 'slab')
                % Skier triggered crack initiation boundary conditions
                if a == 0
                    % Zero crack length
                    paths.beam = 'symbolic/beam/timoshenko/complex/nocrack';
                    args.M = {EI,kGA,kn,l1,l4,lam1,lam2};
                    args.n = {Fn,kGA};
                else
                    % Nonzero crack length
                    paths.beam = 'symbolic/beam/timoshenko/complex/crack';
                    args.M = {EI,kGA,kn,l1,l2,l3,l4,lam1,lam2};
                    args.n = {EI,Fn,kGA,kn,l2,l3,qn};
                end
            end
        elseif kn*EI >= 4*kGA^2
            % Roots of characteristic polynomial are real
            lam1 = sqrt(kn/kGA + sqrt(kn^2/kGA^2 - 4*kn/EI))/sqrt(2);
            lam2 = sqrt(kn/kGA - sqrt(kn^2/kGA^2 - 4*kn/EI))/sqrt(2);
            if strcmp(bc, 'pst')
                % Propagation saw test boundary conditions
                paths.beam = 'symbolic/beam/timoshenko/real/pst';
                args.M = {EI,a,kGA,kn,l,lam1,lam2};
%                 args.n = {EI,a,kGA,kn,qn};
                args.n = {EI,Mr,Vr,a,kGA,kn,qn};
            elseif strcmp(bc, 'slab')
                % Skier triggered crack initiation boundary conditions
                if a == 0
                    % Zero crack length
                    paths.beam = 'symbolic/beam/timoshenko/real/nocrack';
                    args.M = {EI,kGA,kn,l1,l4,lam1,lam2};
                    args.n = {Fn,kGA};
                else
                    % Nonzero crack length
                    paths.beam = 'symbolic/beam/timoshenko/real/crack';
                    args.M = {EI,kGA,kn,l1,l2,l3,l4,lam1,lam2};
                    args.n = {EI,Fn,kGA,kn,l2,l3,qn};
                end
            end
        end
    % Timoshenko beam on Pasternak foundation
    elseif strcmp(kinematics, 'timoshenko-pasternak')
        if (kn*EI)^2 + (ks*kGA)^2 < 2*EI*kn*kGA*(ks + 2*kGA)
            % Roots of characteristic polynomial are complex
            lam1 = sqrt(...
                sqrt(kn*kGA/(4*EI*(ks + kGA))) + ...
                (EI*kn + kGA*ks)/(4*EI*(ks + kGA)));
            lam2 = sqrt(...
                sqrt(kn*kGA/(4*EI*(ks + kGA))) - ...
                (EI*kn + kGA*ks)/(4*EI*(ks + kGA)));
            if strcmp(bc, 'pst')
                % Propagation saw test boundary condtions
                paths.beam = 'symbolic/beam/timoshenkopasternak/complex/pst';
                args.M = {EI,a,kGA,kn,ks,l,lam1,lam2};
%                 args.n = {EI,a,kGA,kn,qn};
                args.n = {EI,Mr,Vr,a,kGA,kn,qn};
            elseif strcmp(bc, 'slab')
                % Skier triggered crack initiation boundary conditions
                if a == 0
                    % Zero crack length
                    paths.beam = 'symbolic/beam/timoshenkopasternak/complex/nocrack';
                    args.M = {EI,kGA,kn,ks,l1,l4,lam1,lam2};
                    args.n = {Fn,kGA};
                else
                    % Nonzero crack length
                    paths.beam = 'symbolic/beam/timoshenkopasternak/complex/crack';
                    args.M = {EI,kGA,kn,ks,l1,l2,l3,l4,lam1,lam2};
                    args.n = {EI,Fn,kGA,kn,l2,l3,qn};
                end
            end
        elseif (kn*EI)^2 + (ks*kGA)^2 >= 2*EI*kn*kGA*(ks + 2*kGA)
            % Roots of characteristic polynomial are real
            lam1 = sqrt(...
                (EI*kn + kGA*ks)/(EI*(ks + kGA)) + ...
                sqrt((EI*kn + kGA*ks)^2/(EI*(ks + kGA))^2 - ...
                     4*kn*kGA/(EI*(ks + kGA))))/sqrt(2);
            lam2 = sqrt(...
                (EI*kn + kGA*ks)/(EI*(ks + kGA)) - ...
                sqrt((EI*kn + kGA*ks)^2/(EI*(ks + kGA))^2 - ...
                     4*kn*kGA/(EI*(ks + kGA))))/sqrt(2);
            if strcmp(bc, 'pst')
                % Propagation saw test boundary conditions
                paths.beam = 'symbolic/beam/timoshenkopasternak/real/pst';
                args.M = {EI,a,kGA,kn,ks,l,lam1,lam2};
%                 args.n = {EI,a,kGA,kn,qn};
                args.n = {EI,Mr,Vr,a,kGA,kn,qn};
            elseif strcmp(bc, 'slab')
                % Skier triggered crack initiation boundary conditions
                if a == 0
                    % Zero crack length
                    paths.beam = 'symbolic/beam/timoshenkopasternak/real/nocrack';
                    args.M = {EI,kGA,kn,ks,l1,l4,lam1,lam2};
                    args.n = {Fn,kGA};
                else
                    % Nonzero crack length
                    paths.beam = 'symbolic/beam/timoshenkopasternak/real/crack';
                    args.M = {EI,kGA,kn,ks,l1,l2,l3,l4,lam1,lam2};
                    args.n = {EI,Fn,kGA,kn,l2,l3,qn};
                end
            end
        end
    end

end

