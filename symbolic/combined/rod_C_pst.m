function C = rod_coefficientmatrix(a,l,mu)
%ROD_COEFFICIENTMATRIX
%    C = ROD_COEFFICIENTMATRIX(A,L,MU)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    01-Mar-2019 00:54:50

t2 = a-l;
t3 = mu.*t2;
t4 = sinh(t3);
t5 = cosh(t3);
C = reshape([0.0,t5,-mu.*t4,0.0,mu,-t4,mu.*t5,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,-1.0,1.0],[4,4]);
