function W = deflection(x,EI,a,c1,c2,c3,c4,c5,c6,c7,c8,kn,l,lam1,lam2,qn)
%DEFLECTION
%    W = DEFLECTION(X,EI,A,C1,C2,C3,C4,C5,C6,C7,C8,KN,L,LAM1,LAM2,QN)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    04-Feb-2019 11:05:54

if ((a+x <= l) & (0.0 <= x))
    W = c1.*cosh(lam1.*x)+c3.*cosh(lam2.*x)+c2.*sinh(lam1.*x)+c4.*sinh(lam2.*x)+qn./kn;
elseif ((l < a+x) & (x <= l))
    W = c5+c7.*(a-l+x).^2+c8.*(a-l+x).^3+c6.*(a-l+x)+(qn.*(a-l+x).^4.*(1.0./2.4e1))./EI;
else
    W = 0.0;
end
