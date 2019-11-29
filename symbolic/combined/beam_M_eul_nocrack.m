function M = beam_coefficientmatrix(l1,l4,lam)
%BEAM_COEFFICIENTMATRIX
%    M = BEAM_COEFFICIENTMATRIX(L1,L4,LAM)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    01-Mar-2019 00:54:47

t2 = lam.^2;
t3 = t2.*2.0;
t4 = lam.*t2.*2.0;
t5 = l1.*lam;
t6 = exp(t5);
t7 = cos(t5);
t8 = exp(-t5);
t9 = sin(t5);
t10 = lam.*t6.*t7;
t11 = lam.*t2.*t7.*t8.*2.0;
t12 = l4.*lam;
t13 = exp(t12);
t14 = sin(t12);
t15 = exp(-t12);
t16 = cos(t12);
t17 = lam.*t2.*t15.*t16.*2.0;
M = reshape([0.0,lam.*t2.*-2.0,t6.*t7,t10-lam.*t6.*t9,t2.*t6.*t9.*-2.0,lam.*t2.*t6.*t7.*-2.0-lam.*t2.*t6.*t9.*2.0,0.0,0.0,t3,t4,t6.*t9,t10+lam.*t6.*t9,t2.*t6.*t7.*2.0,lam.*t2.*t6.*t7.*2.0-lam.*t2.*t6.*t9.*2.0,0.0,0.0,0.0,t4,t7.*t8,-lam.*t7.*t8-lam.*t8.*t9,t2.*t8.*t9.*2.0,t11-lam.*t2.*t8.*t9.*2.0,0.0,0.0,-t3,t4,t8.*t9,lam.*t7.*t8-lam.*t8.*t9,t2.*t7.*t8.*-2.0,t11+lam.*t2.*t8.*t9.*2.0,0.0,0.0,0.0,0.0,-1.0,-lam,0.0,t4,t2.*t13.*t14.*-2.0,lam.*t2.*t13.*t14.*-2.0-lam.*t2.*t13.*t16.*2.0,0.0,0.0,0.0,-lam,-t3,-t4,t2.*t13.*t16.*2.0,lam.*t2.*t13.*t14.*-2.0+lam.*t2.*t13.*t16.*2.0,0.0,0.0,-1.0,lam,0.0,-t4,t2.*t14.*t15.*2.0,t17-lam.*t2.*t14.*t15.*2.0,0.0,0.0,0.0,-lam,t3,-t4,t2.*t15.*t16.*-2.0,t17+lam.*t2.*t14.*t15.*2.0],[8,8]);
