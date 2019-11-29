function M = beam_coefficientmatrix(EI,kGA,kn,l1,l4,lam1,lam2)
%BEAM_COEFFICIENTMATRIX
%    M = BEAM_COEFFICIENTMATRIX(EI,KGA,KN,L1,L4,LAM1,LAM2)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    01-Feb-2019 15:25:02

t2 = lam1.^2;
t3 = lam2.^2;
t4 = 1.0./kGA.^2;
t5 = EI.*kn.*t4;
t6 = t5-1.0;
t7 = 1.0./kGA;
t8 = t2-t3;
t9 = t6.*t8;
t10 = t2.^2;
t11 = t3.^2;
t34 = t2.*t3.*6.0;
t12 = t10+t11-t34;
t35 = EI.*t7.*t12;
t13 = t9-t35;
t14 = lam1.*lam2.*t6.*2.0;
t15 = lam1.*lam2.*t2.*4.0;
t16 = lam1.*t6;
t17 = lam1.*t2;
t18 = lam2.*t6;
t19 = lam2.*t2.*3.0;
t29 = lam2.*t3;
t20 = t19-t29;
t30 = EI.*t7.*t20;
t21 = lam2+t18-t30;
t22 = l1.*lam1;
t23 = exp(t22);
t24 = l1.*lam2;
t25 = cos(t24);
t26 = exp(-t22);
t27 = sin(t24);
t31 = lam1.*t3.*3.0;
t32 = t17-t31;
t28 = EI.*t7.*t32;
t33 = -t18+t30;
t39 = lam1.*lam2.*t3.*4.0;
t40 = t15-t39;
t36 = EI.*t7.*t40;
t37 = -t14+t36;
t38 = -t9+t35;
t41 = lam1.*t23.*t25;
t42 = t41-lam2.*t23.*t27;
t43 = lam2.*t23.*t25;
t44 = lam1.*t23.*t27;
t45 = t43+t44;
t46 = lam1.*t25.*t26;
t47 = lam2.*t26.*t27;
t48 = lam2.*t25.*t26;
t49 = t48-lam1.*t26.*t27;
t50 = l4.*lam1;
t51 = exp(t50);
t52 = l4.*lam2;
t53 = cos(t52);
t54 = sin(t52);
t55 = exp(-t50);
t56 = lam1.*t51.*t53;
t57 = lam2.*t51.*t53;
t58 = lam1.*t51.*t54;
t59 = lam1.*t53.*t55;
t60 = lam2.*t54.*t55;
t61 = lam1.*t54.*t55;
M = reshape([t13,lam1+t16-EI.*t7.*(t17-lam1.*t3.*3.0),t23.*t25,t6.*t42-EI.*t7.*(lam1.*t2.*t23.*t25-lam1.*t3.*t23.*t25.*3.0-lam2.*t2.*t23.*t27.*3.0+lam2.*t3.*t23.*t27),-t6.*(-t2.*t23.*t25+t3.*t23.*t25+lam1.*lam2.*t23.*t27.*2.0)-EI.*t7.*(t10.*t23.*t25+t11.*t23.*t25-t2.*t3.*t23.*t25.*6.0-lam1.*lam2.*t2.*t23.*t27.*4.0+lam1.*lam2.*t3.*t23.*t27.*4.0),t42,0.0,0.0,t14-EI.*t7.*(t15-lam1.*lam2.*t3.*4.0),t21,t23.*t27,t6.*t45-EI.*t7.*(lam2.*t2.*t23.*t25.*3.0+lam1.*t2.*t23.*t27-lam2.*t3.*t23.*t25-lam1.*t3.*t23.*t27.*3.0),t6.*(t2.*t23.*t27-t3.*t23.*t27+lam1.*lam2.*t23.*t25.*2.0)-EI.*t7.*(t10.*t23.*t27+t11.*t23.*t27-t2.*t3.*t23.*t27.*6.0+lam1.*lam2.*t2.*t23.*t25.*4.0-lam1.*lam2.*t3.*t23.*t25.*4.0),t45,0.0,0.0,t13,-lam1-t16+t28,t25.*t26,-t6.*(t46+t47)+EI.*t7.*(lam1.*t2.*t25.*t26-lam1.*t3.*t25.*t26.*3.0+lam2.*t2.*t26.*t27.*3.0-lam2.*t3.*t26.*t27),t6.*(t2.*t25.*t26-t3.*t25.*t26+lam1.*lam2.*t26.*t27.*2.0)-EI.*t7.*(t10.*t25.*t26+t11.*t25.*t26-t2.*t3.*t25.*t26.*6.0+lam1.*lam2.*t2.*t26.*t27.*4.0-lam1.*lam2.*t3.*t26.*t27.*4.0),-t46-t47,0.0,0.0,t37,t21,t26.*t27,t6.*t49-EI.*t7.*(lam2.*t2.*t25.*t26.*3.0-lam1.*t2.*t26.*t27-lam2.*t3.*t25.*t26+lam1.*t3.*t26.*t27.*3.0),-t6.*(-t2.*t26.*t27+t3.*t26.*t27+lam1.*lam2.*t25.*t26.*2.0)-EI.*t7.*(t10.*t26.*t27+t11.*t26.*t27-t2.*t3.*t26.*t27.*6.0-lam1.*lam2.*t2.*t25.*t26.*4.0+lam1.*lam2.*t3.*t25.*t26.*4.0),t49,0.0,0.0,0.0,0.0,-1.0,-t16+t28,t38,-lam1,-t6.*(-t2.*t51.*t53+t3.*t51.*t53+lam1.*lam2.*t51.*t54.*2.0)-EI.*t7.*(t10.*t51.*t53+t11.*t51.*t53-t2.*t3.*t51.*t53.*6.0-lam1.*lam2.*t2.*t51.*t54.*4.0+lam1.*lam2.*t3.*t51.*t54.*4.0),t56+t6.*(t56-lam2.*t51.*t54)-EI.*t7.*(lam1.*t2.*t51.*t53-lam1.*t3.*t51.*t53.*3.0-lam2.*t2.*t51.*t54.*3.0+lam2.*t3.*t51.*t54)-lam2.*t51.*t54,0.0,0.0,0.0,t33,t37,-lam2,t6.*(t2.*t51.*t54-t3.*t51.*t54+lam1.*lam2.*t51.*t53.*2.0)-EI.*t7.*(t10.*t51.*t54+t11.*t51.*t54-t2.*t3.*t51.*t54.*6.0+lam1.*lam2.*t2.*t51.*t53.*4.0-lam1.*lam2.*t3.*t51.*t53.*4.0),t57+t58+t6.*(t57+t58)-EI.*t7.*(lam1.*t2.*t51.*t54+lam2.*t2.*t51.*t53.*3.0-lam1.*t3.*t51.*t54.*3.0-lam2.*t3.*t51.*t53),0.0,0.0,-1.0,t16-t28,t38,lam1,t6.*(t2.*t53.*t55-t3.*t53.*t55+lam1.*lam2.*t54.*t55.*2.0)-EI.*t7.*(t10.*t53.*t55+t11.*t53.*t55-t2.*t3.*t53.*t55.*6.0+lam1.*lam2.*t2.*t54.*t55.*4.0-lam1.*lam2.*t3.*t54.*t55.*4.0),-t59-t60-t6.*(t59+t60)+EI.*t7.*(lam1.*t2.*t53.*t55-lam1.*t3.*t53.*t55.*3.0+lam2.*t2.*t54.*t55.*3.0-lam2.*t3.*t54.*t55),0.0,0.0,0.0,t33,t14-t36,-lam2,-t6.*(-t2.*t54.*t55+t3.*t54.*t55+lam1.*lam2.*t53.*t55.*2.0)-EI.*t7.*(t10.*t54.*t55+t11.*t54.*t55-t2.*t3.*t54.*t55.*6.0-lam1.*lam2.*t2.*t53.*t55.*4.0+lam1.*lam2.*t3.*t53.*t55.*4.0),-t61-t6.*(t61-lam2.*t53.*t55)+EI.*t7.*(lam1.*t2.*t54.*t55-lam2.*t2.*t53.*t55.*3.0-lam1.*t3.*t54.*t55.*3.0+lam2.*t3.*t53.*t55)+lam2.*t53.*t55],[8,8]);
