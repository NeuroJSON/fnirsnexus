function act = readspatialact(fname, trials)
% spatial_act.csv (trial_type, vertex 0-based, chromo, uM) -> [nvert x {HbO,HbR} x ntrial]
T = readtable(fname, 'TextType', 'string');
nv = max(T.vertex) + 1;
chromo = {'HbO', 'HbR'};
act = zeros(nv, 2, numel(trials));
for t = 1:numel(trials)
    for c = 1:2
        sel = T.trial_type == trials{t} & T.chromo == chromo{c};
        act(T.vertex(sel) + 1, c, t) = T.spatial_act(sel);
    end
end
end
