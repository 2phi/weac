% Find critical skier weight for weak layer anti-crack nucleation using FFM
%

%% Clear

close all
clear all

%% User input

% Switches
data.euler_or_timoshenko = 'timoshenko';    % 'euler', 'timoshenko'

% Geometry
data.h = 400;               % Slab height [mm]
data.l = 25*data.h;         % Total beam length [mm]
data.t = 20;                % Weak layer thickness [mm]
data.b = 1;                 % Out-of-plane thickness [mm]
data.phi = 30;              % Slope angle [°]
data.lski = 1000;           % Ski length [mm]

% Material parameters
data.rho = 200;             % Slab density [kg/m^3]
data.Eslab = 4;             % Slab Young's modulus [MPa]
data.Eweak = 0.15;          % Weak layer Young's modulus [MPa]
data.nu = 0.25;             % Poisson's ratios (assumption) [–]

% Fracture properties
data.sigc = 2600;           % Compressive strength [Pa]
data.tauc = 700;            % Shear strength [Pa]
data.Gc = 3.0;              % Fracture toughness [J/m^2]

% Fracture criterion
scrit = 'quads';            % Stress criterion: 'quads', 'mccapconti', ...
gcrit = 'factor20';         % Energy criterion: 'factor10', 'factor20', ...
m0_max = 500;               % Max. initial guess for skier weight

%% Run

% Update data
data = update_data(data);   % Compute additional parameters and convert units

% Compute failure load
[mf, daf] = ffm_find_failure_load(m0_max, data, scrit, gcrit);
