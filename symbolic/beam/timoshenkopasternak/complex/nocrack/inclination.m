function Wp = inclination(x,c1,c2,c3,c4,c13,c14,c15,c16,l,l1,lam1,lam2)
%INCLINATION
%    WP = INCLINATION(X,C1,C2,C3,C4,C13,C14,C15,C16,L,L1,LAM1,LAM2)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    04-Feb-2019 11:05:47

if ((x <= l1) & (0.0 <= x))
    Wp = exp(lam1.*x).*(c2.*lam2.*cos(lam2.*x)-c1.*lam2.*sin(lam2.*x))+exp(-lam1.*x).*(c4.*lam2.*cos(lam2.*x)-c3.*lam2.*sin(lam2.*x))+lam1.*exp(lam1.*x).*(c1.*cos(lam2.*x)+c2.*sin(lam2.*x))-lam1.*exp(-lam1.*x).*(c3.*cos(lam2.*x)+c4.*sin(lam2.*x));
elseif ((l1 < x) & (x <= l))
    Wp = exp(-lam1.*(l1-x)).*(c14.*lam2.*cos(lam2.*(l1-x))+c13.*lam2.*sin(lam2.*(l1-x)))+exp(lam1.*(l1-x)).*(c16.*lam2.*cos(lam2.*(l1-x))+c15.*lam2.*sin(lam2.*(l1-x)))+lam1.*exp(-lam1.*(l1-x)).*(c13.*cos(lam2.*(l1-x))-c14.*sin(lam2.*(l1-x)))-lam1.*exp(lam1.*(l1-x)).*(c15.*cos(lam2.*(l1-x))-c16.*sin(lam2.*(l1-x)));
else
    Wp = 0.0;
end
