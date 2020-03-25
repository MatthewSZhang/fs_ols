function [IND, criteria] = fs_lda(X, y, t)
% Feature Selection by Linear Discriminant Analysis

criteria = zeros(t, 1);
[N, n] = size(X);

IND = zeros(t, 1);
Xs = zeros(N, t);
Xr = X;
INDr = 1:n;
for k = 1:t
    q = size(Xr, 2);
    RXY = zeros(q, 1);
    for s = 1:q
        X_temp = [Xs(:, 1:k-1), Xr(:, s)];
        [~,J] = f_lda(X_temp,y); % J is Fisher's criterion
        R2_temp = J./(1+J);
        RXY(s) = sum(R2_temp);
    end
    [criteria(k), ind] = max(abs(RXY));
    IND(k) = INDr(ind);
    INDr(:, ind) = [];
    Xr(:, ind) = [];
    Xs(:, k) = X(:, IND(k));
end
end