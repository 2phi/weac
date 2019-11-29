function g_crit_val = crit_func_G (Gbari, Gbarii, G_crit, data)
    % G_crit is string defining type of criteiron   
    % energy criterion: 'sum', 'only_modeI', 'only_modeII', 'factor2', 'factor10'
    Gbari = Gbari;
    Gbarii = Gbarii;
    
    if strcmp(G_crit, 'linG')
        g_crit_val = (Gbari + Gbarii)/data.Gc;
    elseif strcmp(G_crit, 'quadG')  
        g_crit_val = (Gbari/data.Gc)^2 + (Gbarii/(data.Gc))^2;
    elseif strcmp(G_crit, 'quadG2')  
        g_crit_val = (Gbari/data.Gc)^2 + (Gbarii/(data.Gc/2.))^2;  % GIIc is factor 2 smaller than GIc
    elseif strcmp(G_crit, 'quadG100')  
        g_crit_val = (Gbari/data.Gc)^2 + (Gbarii/(data.Gc/100.))^2;  % GIIc is factor 100 smaller than GIc
    elseif strcmp(G_crit, 'only mode I')
        g_crit_val = Gbari/data.Gc;
    elseif strcmp(G_crit, 'factor1')   % GIIc is factor 1 smaller than GIc
        g_crit_val = Gbari/data.Gc + Gbarii/(data.Gc/1.);
    elseif strcmp(G_crit, 'factor2')   % GIIc is factor 2 smaller than GIc
        g_crit_val = Gbari/data.Gc + Gbarii/(data.Gc/2.);
    elseif strcmp(G_crit, 'factor10')   % GIIc is factor 10 smaller than GIc
        g_crit_val = Gbari/data.Gc + Gbarii/(data.Gc/10.);
    elseif strcmp(G_crit, 'factor20')   % GIIc is factor 20 smaller than GIc
        g_crit_val = Gbari/data.Gc + Gbarii/(data.Gc/20.);
    elseif strcmp(G_crit, 'factor30')   % GIIc is factor 30 smaller than GIc
        g_crit_val = Gbari/data.Gc + Gbarii/(data.Gc/30.);
    elseif strcmp(G_crit, 'factor40')   % GIIc is factor 40 smaller than GIc
        g_crit_val = Gbari/data.Gc + Gbarii/(data.Gc/40.);
    elseif strcmp(G_crit, 'factor60')   % GIIc is factor 60 smaller than GIc
        g_crit_val = Gbari/data.Gc + Gbarii/(data.Gc/60.);
    elseif strcmp(G_crit, 'factor100')   % GIIc is factor 100 smaller than GIc
        g_crit_val = Gbari/data.Gc + Gbarii/(data.Gc/10.);
    elseif strcmp(G_crit, 'expo_n')
        g_crit_val = ((Gbari/data.GIc).^data.n + (Gbarii/data.GIIc).^data.n).^(1./data.n);
    elseif strcmp(G_crit, 'expo_mn')
        g_crit_val = ((Gbari/data.GIc).^data.n + (Gbarii/data.GIIc).^data.m).^(1.);
    elseif strcmp(G_crit, 'BK')
        g_crit_val = (Gbari + Gbarii)./ ... 
                 (data.GIc - (data.GIIc-data.GIc).*((Gbarii)./(Gbari + Gbarii)).^data.eta);
    else
        warning('Energy criterion not defined')
  end
end