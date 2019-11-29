function W = deflection(x,c1,c2,c3,c4,c13,c14,c15,c16,kn,l,l1,lam1,lam2,qn)
%DEFLECTION
%    W = DEFLECTION(X,C1,C2,C3,C4,C13,C14,C15,C16,KN,L,L1,LAM1,LAM2,QN)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    01-Feb-2019 15:25:07

if ((x <= l1) & (0.0 <= x))
    W = c1.*cosh(lam1.*x)+c3.*cosh(lam2.*x)+c2.*sinh(lam1.*x)+c4.*sinh(lam2.*x)+qn./kn;
elseif ((l1 < x) & (x <= l))
    W = c13.*cosh(lam1.*(l1-x))+c15.*cosh(lam2.*(l1-x))+qn./kn-c14.*sinh(lam1.*(l1-x))-c16.*sinh(lam2.*(l1-x));
else
    W = 0.0;
end
