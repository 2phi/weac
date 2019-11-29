function Wp = inclination(x,EI,c1,c2,c3,c4,c6,c7,c8,c10,c11,c12,c13,c14,c15,c16,l,l1,l2,l3,lam1,lam2,qn)
%INCLINATION
%    WP = INCLINATION(X,EI,C1,C2,C3,C4,C6,C7,C8,C10,C11,C12,C13,C14,C15,C16,L,L1,L2,L3,LAM1,LAM2,QN)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    04-Feb-2019 11:05:49

if ((x <= l1) & (0.0 <= x))
    Wp = exp(lam1.*x).*(c2.*lam2.*cos(lam2.*x)-c1.*lam2.*sin(lam2.*x))+exp(-lam1.*x).*(c4.*lam2.*cos(lam2.*x)-c3.*lam2.*sin(lam2.*x))+lam1.*exp(lam1.*x).*(c1.*cos(lam2.*x)+c2.*sin(lam2.*x))-lam1.*exp(-lam1.*x).*(c3.*cos(lam2.*x)+c4.*sin(lam2.*x));
elseif ((x <= l1+l2) & (l1 < x))
    Wp = c6-c7.*(l1-x).*2.0+c8.*(l1-x).^2.*3.0-(qn.*(l1-x).^3.*(1.0./6.0))./EI;
elseif ((l1+l2 < x) & (x <= l1+l2+l3))
    Wp = c10+c12.*(l1+l2-x).^2.*3.0-c11.*(l1+l2-x).*2.0-(qn.*(l1+l2-x).^3.*(1.0./6.0))./EI;
elseif ((x <= l) & (l1+l2+l3 < x))
    Wp = exp(-lam1.*(l1+l2+l3-x)).*(c14.*lam2.*cos(lam2.*(l1+l2+l3-x))+c13.*lam2.*sin(lam2.*(l1+l2+l3-x)))+exp(lam1.*(l1+l2+l3-x)).*(c16.*lam2.*cos(lam2.*(l1+l2+l3-x))+c15.*lam2.*sin(lam2.*(l1+l2+l3-x)))+lam1.*exp(-lam1.*(l1+l2+l3-x)).*(c13.*cos(lam2.*(l1+l2+l3-x))-c14.*sin(lam2.*(l1+l2+l3-x)))-lam1.*exp(lam1.*(l1+l2+l3-x)).*(c15.*cos(lam2.*(l1+l2+l3-x))-c16.*sin(lam2.*(l1+l2+l3-x)));
else
    Wp = 0.0;
end
