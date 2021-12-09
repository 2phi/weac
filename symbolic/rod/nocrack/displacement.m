function U = displacement(x,kt,l,l1,mu,qt,z1,z2,z7,z8)
%DISPLACEMENT
%    U = DISPLACEMENT(X,KT,L,L1,MU,QT,Z1,Z2,Z7,Z8)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    01-Mar-2019 00:54:50

if ((x <= l1) & (0.0 <= x))
    U = z1.*cosh(mu.*x)+z2.*sinh(mu.*x)+qt./kt;
elseif ((l1 < x) & (x <= l))
    U = qt./kt+z7.*cosh(mu.*(l1-x))-z8.*sinh(mu.*(l1-x));
else
    U = 0.0;
end