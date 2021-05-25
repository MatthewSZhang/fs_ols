function [IND, criteria] = fs_ols(X, Y, t)
% Feature Selection by Orthogonal Least Squares
% Input:
% X: N * n matrix. N observations and n features. Feature matrix.
% Y: N * c-1 matrix. N observations and c classes. c-1 dummy encoded response.
% t: The number of features required to be selected.

% Output
IND = zeros(t, 1);
criteria = zeros(t, 1);

n = size(X, 2);
INDr = 1:n; % index of rest features
N = size(Y, 1); % number of the observations

% Step 1
Yc = Y-mean(Y); % Yc
Vc = f_orth(Yc);

% Step 2
XCr = X - mean(X);

% Step 3
WCs = zeros(N, t);
WCr = XCr./vecnorm(XCr);
for k = 1:t
    % Step 4
    g = WCr'*Vc;
    h = g.^2; % squared orthogonal correlation
    RwV = sum(h, 2); % multiple correlation
    
    % Step 5
    [criteria(k), ind] = max(RwV);
    IND(k) = INDr(ind);
    INDr(:, ind) = [];

    % Step 3
    WCs(:, k) = WCr(:, ind);

    WCr(:, ind) = [];
    R = WCs(:, k)'*WCr;
    WCr = WCr - WCs(:, k).*R;
    WCr = WCr./vecnorm(WCr);
end

end


function [Q, R] = f_orth(X)
% Orthogonalisation of a Matrix by the Modified Gram-Schmidt
[N, n] = size(X);
Q = zeros(N, n);
R = zeros(n);
for k = 1:n
    Q(:, k) = X(:, k);
    for s = 1:k-1
        R(s, k) = Q(:, s)'*Q(:, k);
        Q(:, k) = Q(:, k) - R(s, k)*Q(:, s);
    end
    R(k, k) = norm(Q(:, k));
    Q(:, k) = Q(:, k)/R(k, k);
end
end


