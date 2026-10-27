function E = prahl(wl)
% Prahl molar extinction (cm^-1/M, base 10), as tabulated in cedalion's
% prahl_absorption_spectrum.tsv; Cedalion converts with ln(10)/10 -> 1/(mm M)
tab = [760, 586, 1548.52; ...
       850, 1058, 691.32];
[ok, loc] = ismember(wl(:), tab(:, 1));
if ~all(ok)
    error('add the Prahl coefficients for %s nm', mat2str(wl(~ok)));
end
E = log(10) * tab(loc, 2:3) / 10;
end
