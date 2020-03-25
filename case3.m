clear
clc

addpath('C:\Users\SZhang\Google Drive\Matlab\20190725 OLS\mRMR_0.9_compiled')
addpath('C:\Users\SZhang\Google Drive\Matlab\20190725 OLS\mRMR_0.9_compiled\mi')


Nf = 100; % Library contains Nf features
N = 900; % Library contains N samples
Ntimes = 100;

N_ols = 0;
N_mrmr_d0 = 0;
N_mrmr_q0 = 0;
N_mrmr_d1 = 0;
N_mrmr_q1 = 0;
N_mrmr_d2 = 0;
N_mrmr_q2 = 0;
N_mrmr_d3 = 0;
N_mrmr_q3 = 0;
N_mrmr_d4 = 0;
N_mrmr_q4 = 0;


for t = 1:Ntimes
%% Generate X and Y
rng(t)

% Generate X
mu = randn(1, Nf); % Generate normal distributed random numbers for mean values
sigma = wishrnd(diag(rand(1, Nf)),N)./N; % Sample a positive definite matrix from Wishart Distribution
X_all = mvnrnd(mu,sigma, N); % Generate multivariate normal random numbers


% Generate Response Y
findex_real = [5 10 15]; % Real index of features
X = X_all(:,findex_real); % Binomial responses using just 3 of the predictors plus a constant
b1 = [0; -1; -1; 1]; % First number is intercept, and others are coefficients
b2 = [0; 1; -1; -1];

prob_ratio_13 = exp(X*b1(2:end)+b1(1));
prob_ratio_23 = exp(X*b2(2:end)+b2(1));
prob3 = 1./(1 + prob_ratio_13 + prob_ratio_23);
prob1 = prob3.*prob_ratio_13;
prob2 = prob3.*prob_ratio_23;
prob = [prob1, prob2, prob3];

Y = mnrnd(1,prob); % Multinomial response Y

y = zeros(N, 1);
for p = 1:N
    y(p) = find(Y(p, :)); % generate label for mRMR
end

Y = Y(:, 2:end); % change from c dummy encoding to c-1 dummy encoding

%% Feature selection
% OLS
[IND_ols, criteria] = fs_ols(X_all, Y, 3); % OLS


% mRMR
X_all_d = zeros(N, Nf);
for p = 1:Nf
    edges = [-inf, mean(X_all(:, p)), inf];
    X_all_d(:, p) = discretize(X_all(:, p),edges);
end
IND_mrmr_d0 = mrmr_mid_d(X_all_d, y, 3);
IND_mrmr_q0 = mrmr_miq_d(X_all_d, y, 3);


X_all_d = zeros(N, Nf);
for p = 1:Nf
    edges = [-inf, std(X_all(:, p))*(-1:1)+mean(X_all(:, p)), inf];
    X_all_d(:, p) = discretize(X_all(:, p),edges);
end
IND_mrmr_d1 = mrmr_mid_d(X_all_d, y, 3);
IND_mrmr_q1 = mrmr_miq_d(X_all_d, y, 3);


X_all_d = zeros(N, Nf);
for p = 1:Nf
    edges = [-inf, std(X_all(:, p))*(-2:2)+mean(X_all(:, p)), inf];
    X_all_d(:, p) = discretize(X_all(:, p),edges);
end
IND_mrmr_d2 = mrmr_mid_d(X_all_d, y, 3);
IND_mrmr_q2 = mrmr_miq_d(X_all_d, y, 3);


X_all_d = zeros(N, Nf);
for p = 1:Nf
    edges = [-inf, std(X_all(:, p))*(-3:3)+mean(X_all(:, p)), inf];
    X_all_d(:, p) = discretize(X_all(:, p),edges);
end
IND_mrmr_d3 = mrmr_mid_d(X_all_d, y, 3);
IND_mrmr_q3 = mrmr_miq_d(X_all_d, y, 3);


IND_mrmr_d4 = mrmr_mid_d(X_all, y, 3);
IND_mrmr_q4 = mrmr_miq_d(X_all, y, 3);


%% Count right times
if isequal(sort(IND_ols)', findex_real)
    N_ols = N_ols + 1;
end


if isequal(sort(IND_mrmr_d0), findex_real)
    N_mrmr_d0 = N_mrmr_d0 + 1;
end
if isequal(sort(IND_mrmr_q0), findex_real)
    N_mrmr_q0 = N_mrmr_q0 + 1;
end
if isequal(sort(IND_mrmr_d1), findex_real)
    N_mrmr_d1 = N_mrmr_d1 + 1;
end
if isequal(sort(IND_mrmr_q1), findex_real)
    N_mrmr_q1 = N_mrmr_q1 + 1;
end
if isequal(sort(IND_mrmr_d2), findex_real)
    N_mrmr_d2 = N_mrmr_d2 + 1;
end
if isequal(sort(IND_mrmr_q2), findex_real)
    N_mrmr_q2 = N_mrmr_q2 + 1;
end
if isequal(sort(IND_mrmr_d3), findex_real)
    N_mrmr_d3 = N_mrmr_d3 + 1;
end
if isequal(sort(IND_mrmr_q3), findex_real)
    N_mrmr_q3 = N_mrmr_q3 + 1;
end
if isequal(sort(IND_mrmr_d4), findex_real)
    N_mrmr_d4 = N_mrmr_d4 + 1;
end
if isequal(sort(IND_mrmr_q4), findex_real)
    N_mrmr_q4 = N_mrmr_q4 + 1;
end


end

