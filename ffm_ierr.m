function [Gbari, Gbarii] = ffm_ierr(m, a, unitload, dat)
% Compute incremental energy release rate for load m [kg] and crack a [mm]

%% Displacements of uncracked configuration

w0 = @(x) m*ppval(unitload.w, x) + dat.qn/dat.kn;
u0 = @(x) m*ppval(unitload.u, x) + dat.qt/dat.kt;

%% Displacements of cracked configuration

% Get data struct for crack a[mm] and load m [kg]
dat = ffm_geometry_and_loading(m, a, dat);

% Compute beam solution
[args, paths] = slab_stress_fromfile(dat);

% Function handles for displacements for crack a [mm] and load m [kg]
w1 = @(x) deflection(x, args.w{:});
u1 = @(x) displacement(x, args.u{:});

%% Compute incremental energy release rate

% Function values on integration interval
xq = linspace((dat.l - a)/2, (dat.l + a)/2, (dat.l - a)/dat.dx);
w0w1 = w0(xq).*arrayfun(w1, xq);
u0u1 = u0(xq).*arrayfun(u1, xq);

% Integrate [trapz(X, Y, <dimension along which to integrate>)]
Gbari  = dat.kn*trapz(xq, w0w1, 2)/(2*dat.b*a);
Gbarii = dat.kt*trapz(xq, u0u1, 2)/(2*dat.b*a);

% Clean up
rmpath(paths.rod, paths.beam)

end









