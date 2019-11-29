function n = beam_righthandsidevector(EI,Fn,kGA,kn,l2,l3,qn)
%BEAM_RIGHTHANDSIDEVECTOR
%    N = BEAM_RIGHTHANDSIDEVECTOR(EI,FN,KGA,KN,L2,L3,QN)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    01-Feb-2019 15:25:08

t2 = l2.^2;
t3 = 1.0./kGA;
t4 = 1.0./EI;
t5 = 1.0./kn;
t6 = l3.^2;
n = [0.0;0.0;-qn.*t5;0.0;0.0;-qn.*t3;qn.*t2.^2.*t4.*(-1.0./2.4e1);Fn.*t3-l2.*qn.*t2.*t4.*(1.0./6.0);l2.*qn.*t3+l2.*qn.*t2.*t4.*(1.0./6.0);qn.*t2.*t4.*(1.0./2.0);qn.*t5-qn.*t4.*t6.^2.*(1.0./2.4e1);l3.*qn.*t4.*t6.*(-1.0./6.0);l3.*qn.*t3+l3.*qn.*t4.*t6.*(1.0./6.0);qn.*t3+qn.*t4.*t6.*(1.0./2.0);0.0;0.0];
