function [IND, criteria] = fs_cca_d(X, Y, t)
% Feature Selection by Canonical Correlation Analysis
% Feature can be continuous and categorical (discrete).

criteria = zeros(t, 1);
[~, n] = size(X);

IND = zeros(t, 1);
Xs = cell(1, t);
Xr = X;
INDr = 1:n;
for k = 1:t
    q = size(Xr, 2);
    RXY = zeros(q, 1);
    for s = 1:q
        X_temp = cell2mat([Xs(1:k-1), Xr(s)]);
        [~,~,R_temp,~,~] = canoncorr(X_temp,Y);
        RXY(s) = sum(R_temp.^2);
    end
    [criteria(k), ind] = max(abs(RXY));
    IND(k) = INDr(ind);
    INDr(:, ind) = [];
    Xr(ind) = [];
    Xs(k) = X(IND(k));
end
end