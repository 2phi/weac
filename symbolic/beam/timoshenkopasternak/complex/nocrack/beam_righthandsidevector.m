function n = beam_righthandsidevector(Fn,kGA)
%BEAM_RIGHTHANDSIDEVECTOR
%    N = BEAM_RIGHTHANDSIDEVECTOR(FN,KGA)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    04-Feb-2019 11:05:46

n = [0.0;0.0;0.0;0.0;0.0;Fn./kGA;0.0;0.0];