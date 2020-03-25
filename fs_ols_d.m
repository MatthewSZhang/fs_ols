function [IND, criteria] = fs_ols_d(X, Y, t)
% Feature Selection by Orthogonal Least Squares
% Feature can be continuous and categorical (discrete).
% Input:
% X: 1 * n cell. The i^th cell is a N * zi matrix, which is c-1 dummy encoded feature. N observations and n features. Feature matrix.
% Y: N * c-1 matrix. N observations and c classes. c-1 dummy encoded response.
% t: The number of features required to be selected.



% Output
IND = zeros(t, 1);
criteria = zeros(t, 1);

n = size(X, 2);
INDr = 1:n; % index of rest features

% Step 1
Yc = Y-mean(Y); % Yc
Vc = f_orth(Yc);

% Step 2
XCr = cell(1, n);
for k = 1:n
    XCr{k} = X{k} - mean(X{k});
end

% Step 3
WCs = cell(1, t);
WCr = cell(1, n);
for k = 1:n
    WCr{k} = f_orth(XCr{k});
end
q = n;

for k = 1:t
    % Step 4
    RwV = zeros(q, 1);
    for f = 1:q
        g = WCr{f}'*Vc;
        h = g.^2; % squared orthogonal correlation
        RwV(f) = sum(h, 'all'); % multiple correlation
    end

    
    % Step 5
    [criteria(k), ind] = max(RwV);
    IND(k) = INDr(ind);
    INDr(:, ind) = [];

    % Step 3
    WCs(k) = WCr(ind);

    WCr(ind) = [];
    q = q-1;
    
    for f = 1:q
        R = WCs{k}'*WCr{f};
        WCr{f} = WCr{f} - WCs{k}*R;
        WCr{f} = f_orth(WCr{f});
    end
    
end
end

function [Q, R] = f_orth(X)
% Orthogonalisation of a Matrix by the Classical Gram-Schmidt
[N, p] = size(X);
Q = zeros(N, p);
R = zeros(p);
for k = 1:p
    Q(:, k) = X(:, k);
    for s = 1:k-1
        R(s, k) = Q(:, s)'*Q(:, k);
        Q(:, k) = Q(:, k) - R(s, k)*Q(:, s);
    end
    R(k, k) = norm(Q(:, k));
    Q(:, k) = Q(:, k)/R(k, k);
end
end




