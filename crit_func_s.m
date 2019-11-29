function [fstress] = crit_func_s (data, s_crit, sig, tau)
    switch s_crit
        case 'quads'
            fstress = @(x) sqrt((sig(x)/data.sigc)^2 + (tau(x)/data.tauc)^2) - 1;
        case 'mccap_conti'
            fstress = @(x) fstress_mccap_norm(sig(x), tau(x), data.sigc, tauc_mccapconti(sig(x)), 'mccap_inf');
        case 'mc' % uncapped Mohr-Coulomb
            fstress = @(x) nonanfrac(abs(tau(x)), tauc_mc(sig(x)), 0) - 1;
        case 'cap' % only cap of Mohr-Coulomb
            fstress = @(x) nonanfrac(abs(tau(x)), tauc_cap(sig(x)), 0) - 1;
        case 'sig' % only normal stress
            fstress = @(x) -min([sig(x), 0])/data.sigc - 1;
        otherwise % mccap_inf, mccap_lin, mccap_quad
            fstress = @(x) fstress_mccap_norm(sig(x), tau(x), data.sigc, tauc_mccap(sig(x)), s_crit);
    end
    
end