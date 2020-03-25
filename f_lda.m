function [W, Lambda, Sb, Sw] = f_lda(X, y)
L = unique(y);
Nc = length(L);
N = min(size(X, 2), Nc-1);

Sw = 0;
Sb = 0;
m = mean(X);
for p = 1:Nc
    L_mask = y == L(p);
    nL = sum(L_mask);
    Xi = X(L_mask, :);
    Mi = mean(X(L_mask, :));
    Sb = Sb + nL.*((Mi-m)'*(Mi-m));
    Sw = Sw + ((Xi-Mi)'*(Xi-Mi));
end

[W, Lambda]=eig(Sb,Sw);
[Lambda, ind] = sort(diag(Lambda), 'descend');
W = W(:,ind);
Lambda = Lambda(1:N);
W = W(:, 1:N);
% W = W./sqrt(diag(W'*(Sb+Sw)*W))'; % normalisation W to make W'*St*W == 1;
end