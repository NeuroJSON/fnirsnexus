% driver: MCX Jacobian forward (cached fluences; ~35 min on an RTX 5090) -> HRF measurement
try
    mcx_jacobian();                                   % -> mcx_jacobian.mat/.jdb, fluence/
    mcx_build_hrf('mcxfile', fullfile(fileparts(mfilename('fullpath')), 'mcx_jacobian.mat'));
catch ME
    fprintf('ERR: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf('  at %s line %d\n', ME.stack(k).name, ME.stack(k).line);
    end
end
exit;
