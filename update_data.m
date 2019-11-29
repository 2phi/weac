function data = update_data (data)
  % Boundary conditions
    data.bctype = 'slab';                       % 'pst', 'slab'

    % Loading
    data.g = 9.81;                              % Gravitational acceleration [m/s^2]

    % Geometry
    data.A = data.b*data.h;                     % Slab cross section [mm^2]
    data.I = data.b*data.h^3/12;                % Geometrical moment of inertia of slab [mm^4]

    % Material parameters
    data.Gslab = data.Eslab/(2*(1+data.nu));                % Slab yshear modulus [N/mm^2]
    data.kn = data.Eweak*data.b/data.t;                     % Weak layer compressive stiffness [N/mm^3]
    data.kt = data.kn/(2*(1 + data.nu));                    % Weak layer horizontal shear stiffness [N/mm^3]
    data.ks = data.Eweak*data.t*data.b/(6*(1 + data.nu));   % Weak layer vertical shear stiffness [N]
    data.lam = (data.kn/(data.Eslab*data.I))^(1/4)/sqrt(2); % Eigenvalue of beam solution [mm^-1]
    data.mu = sqrt(data.kt/(data.Eslab*data.A));            % Eigenvalue of tensile rod solution [mm^-1]
    data.EI = data.Eslab*data.I;                            % Bending stiffness [Nmm^2]
    data.EA = data.Eslab*data.A;                            % Extension stiffness [N]
    data.kGA = 5/6*data.Gslab*data.A;                       % Shear stiffness [N]

    % Fracture properties
    data.sigc = 1e-6*data.sigc;                 % Convert sigc to [MPa]
    data.tauc = 1e-6*data.tauc;                 % Convert sigc to [MPa]
    data.Gc = 1e-3*data.Gc;                     % Convert Gc to [N/mm = kJ/m^2]

    % Loading
    data.g = data.g*1e3;                        % Convert g to [mm/s^2]
    data.rho = convert_rho(data.rho);           % Convert rho to [t/mm^3]
    data.q = data.rho*data.g*data.h*data.b;     % Line load from slab weigth [N/mm]
    data.qn = data.q*cosd(data.phi);            % Normal (compressive) line load [N/mm]
    data.qt = data.q*sind(data.phi);            % Tangential (shear) line load [N/mm]

    % Misc
    data.dx = 10;                               % Interpolation spacing
    data.Mr = 0;                                % PST only
    data.Vr = 0;                                % PST only
    data.Nr = 0;                                % PST only
end

