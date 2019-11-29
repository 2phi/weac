function data = ffm_geometry_and_loading(m, a, data)
% m in kg
% a in mm

% Geometry
data.a = a;                                 % Crack length [mm]
data.l1 = (data.l - data.a)/2;              % Length of left bedded beam segment [mm]
data.l2 = data.a/2;                         % Length of left free beam segment [mm]
data.l3 = data.a/2;                         % Length of right free beam segment [mm]
data.l4 = (data.l - data.a)/2;              % Length of right bedded beam segment [mm]

% Loading
data.m = m*1e-3;                            % Convert m to [t]
data.F = data.m*data.g*data.b/data.lski;    % Skier load [N]

% Split loading into normal and tangential components
data.Fn = data.F*cosd(data.phi);            % Skier normal (compressive) force [N]
data.Ft = data.F*sind(data.phi);            % Skier tangential (shear) force [N]

end

