function d = rod_righthandsidevector(EA,Nr,a,kt,qt)
%ROD_RIGHTHANDSIDEVECTOR
%    D = ROD_RIGHTHANDSIDEVECTOR(EA,NR,A,KT,QT)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    01-Mar-2019 00:54:51

t2 = 1.0./EA;
d = [0.0;-qt./kt;0.0;Nr.*t2+a.*qt.*t2];
