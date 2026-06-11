try
    s4_recon_surface(0.003, 0.01);   % best settings -> writes recon.bgii
    render_lateral;                  % lateral views from recon.bgii (now the s4 result)
catch ME
    fprintf('ERR: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf('  at %s line %d\n', ME.stack(k).name, ME.stack(k).line);
    end
end
exit;
