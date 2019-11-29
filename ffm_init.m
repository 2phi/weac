function unitload = ffm_init(dat)
% Compute and interpolate stress solution for unit load

%% Compute solution

% Get data struct for no crack and unit load (1 kg)
dat = ffm_geometry_and_loading(1, 0, dat);

% Compute beam solution
[args, paths] = slab_stress_fromfile(dat);

%% Get displacements

% Substitute constants into ansatz and make function handles for unit load
w0 = @(x) deflection(x, args.w{:});
u0 = @(x) displacement(x, args.u{:}); 

% Displacements for unit load only (no gravity load)
wf = @(x) w0(x) - dat.qn/dat.kn;
uf = @(x) u0(x) - dat.qt/dat.kt;

% Scaled displacements for weight m0 in kg with gravity load
% w = @(x) m0*wf(x) + dat.qn/dat.kn;
% u = @(x) m0*uf(x) + dat.qt/dat.kt;

%% Compute stresses

% Stress solutions for unit force load and gravity load
sig0 = @(x) -dat.kn*w0(x)/dat.b;
tau0 = @(x) dat.kt*u0(x)/dat.b;

% Stress solutions for unit load only (no gravity load), sig0 < 0, tau0 > 0
sigf = @(x) sig0(x) + dat.qn/dat.b;
tauf = @(x) tau0(x) - dat.qt/dat.b;

% Scaled stress solutions for weight m0 in kg with gravity load
% sig = @(x) m0*sigf(x) - dat.qn/dat.b;
% tau = @(x) m0*tauf(x) + dat.qt/dat.b;

%% Interpolate

% Coordinates (dat.dx defines spacing)
xq = linspace(0, dat.l, dat.l/dat.dx);
unitload.x = xq;
method = 'spline';

% Interpolate unit load displacements
%unitload.w = griddedInterpolant(xq, arrayfun(wf, xq), method);
unitload.w = interp1(xq, arrayfun(wf, xq), 'spline', 'pp');
%unitload.u = griddedInterpolant(xq, arrayfun(uf, xq), method);
unitload.u = interp1(xq, arrayfun(uf, xq), 'spline', 'pp');

% Interpolate unit load stresses
%unitload.sig = griddedInterpolant(xq, arrayfun(sigf, xq), method);
unitload.sig = interp1(xq, arrayfun(sigf, xq), 'spline', 'pp');
%unitload.tau = griddedInterpolant(xq, arrayfun(tauf, xq), method);
unitload.tau = interp1(xq, arrayfun(tauf, xq), 'spline', 'pp');

%% Misc

% Clean up
rmpath(paths.rod, paths.beam)

end









