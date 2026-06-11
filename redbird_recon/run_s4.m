cd('/home/users/fangq/space/git/Temp/imagespace/redbird_recon');
try
    s4_recon_surface();
catch ME
    fprintf('ERR: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf('  at %s line %d\n', ME.stack(k).name, ME.stack(k).line);
    end
end
exit
