function W = deflection(x,EI,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,kn,l,l1,l2,l3,lam1,lam2,qn)
%DEFLECTION
%    W = DEFLECTION(X,EI,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,KN,L,L1,L2,L3,LAM1,LAM2,QN)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    04-Feb-2019 11:05:53

if ((x <= l1) & (0.0 <= x))
    W = c1.*cosh(lam1.*x)+c3.*cosh(lam2.*x)+c2.*sinh(lam1.*x)+c4.*sinh(lam2.*x)+qn./kn;
elseif ((x <= l1+l2) & (l1 < x))
    W = c5-c6.*(l1-x)+c7.*(l1-x).^2-c8.*(l1-x).^3+(qn.*(l1-x).^4.*(1.0./2.4e1))./EI;
elseif ((l1+l2 < x) & (x <= l1+l2+l3))
    W = c9+c11.*(l1+l2-x).^2-c12.*(l1+l2-x).^3-c10.*(l1+l2-x)+(qn.*(l1+l2-x).^4.*(1.0./2.4e1))./EI;
elseif ((x <= l) & (l1+l2+l3 < x))
    W = c13.*cosh(lam1.*(l1+l2+l3-x))+c15.*cosh(lam2.*(l1+l2+l3-x))-c14.*sinh(lam1.*(l1+l2+l3-x))-c16.*sinh(lam2.*(l1+l2+l3-x))+qn./kn;
else
    W = 0.0;
end
