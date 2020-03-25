% Fisher's Iris
clear
clc

load fisheriris

X = meas([1, 2, 51, 52, 101, 102, 103], :);
Y = [1, 0;
     1, 0;
     0, 1;
     0, 1;
     0, 0;
     0, 0;
     0, 0];
 
%% Step 1
Yc = Y - mean(Y);
Vc1 = Yc(:, 1);
Vc2 = Yc(:, 2) - Yc(:, 2)'*Vc1./(Vc1'*Vc1)*Vc1;

%% Step 2
Xs = [];
Xr = X;

Xcs = Xs - mean(Xs);
Xcr = Xr - mean(Xr);

%% Step 3
Wcr = Xcr;

%% Step 4
h11 = Vc1'*Wcr(:, 1)*Wcr(:, 1)'*Vc1./(Wcr(:, 1)'*Wcr(:, 1)*(Vc1'*Vc1));
h12 = Vc2'*Wcr(:, 1)*Wcr(:, 1)'*Vc2./(Wcr(:, 1)'*Wcr(:, 1)*(Vc2'*Vc2));
h21 = Vc1'*Wcr(:, 2)*Wcr(:, 2)'*Vc1./(Wcr(:, 2)'*Wcr(:, 2)*(Vc1'*Vc1));
h22 = Vc2'*Wcr(:, 2)*Wcr(:, 2)'*Vc2./(Wcr(:, 2)'*Wcr(:, 2)*(Vc2'*Vc2));
h31 = Vc1'*Wcr(:, 3)*Wcr(:, 3)'*Vc1./(Wcr(:, 3)'*Wcr(:, 3)*(Vc1'*Vc1));
h32 = Vc2'*Wcr(:, 3)*Wcr(:, 3)'*Vc2./(Wcr(:, 3)'*Wcr(:, 3)*(Vc2'*Vc2));
h41 = Vc1'*Wcr(:, 4)*Wcr(:, 4)'*Vc1./(Wcr(:, 4)'*Wcr(:, 4)*(Vc1'*Vc1));
h42 = Vc2'*Wcr(:, 4)*Wcr(:, 4)'*Vc2./(Wcr(:, 4)'*Wcr(:, 4)*(Vc2'*Vc2));

R2wcrVc1 = h11 + h12;
R2wcrVc2 = h21 + h22;
R2wcrVc3 = h31 + h32;
R2wcrVc4 = h41 + h42;

[IND_ols, criteria] = fs_ols(X, Y, 3);

%% Step 5
Xs = X(:, 3);
Xr = X(:, [1, 2, 4]);

%% Step 2
Xcs = Xs - mean(Xs);
Xcr = Xr - mean(Xr);

%% Step 3
Wcs = Xcs;
Wcr = [];
Wcr(:, 1) = Xcr(:, 1) - Xcr(:, 1)'*Wcs./(Wcs'*Wcs)*Wcs;
Wcr(:, 2) = Xcr(:, 2) - Xcr(:, 2)'*Wcs./(Wcs'*Wcs)*Wcs;
Wcr(:, 3) = Xcr(:, 3) - Xcr(:, 3)'*Wcs./(Wcs'*Wcs)*Wcs;


%% Step 4
h11 = Vc1'*Wcr(:, 1)*Wcr(:, 1)'*Vc1./(Wcr(:, 1)'*Wcr(:, 1)*(Vc1'*Vc1));
h12 = Vc2'*Wcr(:, 1)*Wcr(:, 1)'*Vc2./(Wcr(:, 1)'*Wcr(:, 1)*(Vc2'*Vc2));
h21 = Vc1'*Wcr(:, 2)*Wcr(:, 2)'*Vc1./(Wcr(:, 2)'*Wcr(:, 2)*(Vc1'*Vc1));
h22 = Vc2'*Wcr(:, 2)*Wcr(:, 2)'*Vc2./(Wcr(:, 2)'*Wcr(:, 2)*(Vc2'*Vc2));
h31 = Vc1'*Wcr(:, 3)*Wcr(:, 3)'*Vc1./(Wcr(:, 3)'*Wcr(:, 3)*(Vc1'*Vc1));
h32 = Vc2'*Wcr(:, 3)*Wcr(:, 3)'*Vc2./(Wcr(:, 3)'*Wcr(:, 3)*(Vc2'*Vc2));



R2wcrVc1 = h11 + h12;
R2wcrVc2 = h21 + h22;
R2wcrVc3 = h31 + h32;

%% Step 5
Xs = X(:, [3, 4]);
Xr = X(:, [1, 2]);

%% Step 2
Xcs = Xs - mean(Xs);
Xcr = Xr - mean(Xr);

%% Step 3
Wcs = [];
Wcs(:, 1) = Xcs(:, 1);
Wcs(:, 2) = Xcs(:, 2) - Xcs(:, 2)'*Wcs(:, 1)./(Wcs(:, 1)'*Wcs(:, 1))*Wcs(:, 1);

Wcr = [];
Wcr(:, 1) = Xcr(:, 1) - Xcr(:, 1)'*Wcs(:, 1)./(Wcs(:, 1)'*Wcs(:, 1))*Wcs(:, 1) - Xcr(:, 1)'*Wcs(:, 2)./(Wcs(:, 2)'*Wcs(:, 2))*Wcs(:, 2);
Wcr(:, 2) = Xcr(:, 2) - Xcr(:, 2)'*Wcs(:, 1)./(Wcs(:, 1)'*Wcs(:, 1))*Wcs(:, 1) - Xcr(:, 2)'*Wcs(:, 2)./(Wcs(:, 2)'*Wcs(:, 2))*Wcs(:, 2);

%% Step 4
h11 = Vc1'*Wcr(:, 1)*Wcr(:, 1)'*Vc1./(Wcr(:, 1)'*Wcr(:, 1)*(Vc1'*Vc1));
h12 = Vc2'*Wcr(:, 1)*Wcr(:, 1)'*Vc2./(Wcr(:, 1)'*Wcr(:, 1)*(Vc2'*Vc2));
h21 = Vc1'*Wcr(:, 2)*Wcr(:, 2)'*Vc1./(Wcr(:, 2)'*Wcr(:, 2)*(Vc1'*Vc1));
h22 = Vc2'*Wcr(:, 2)*Wcr(:, 2)'*Vc2./(Wcr(:, 2)'*Wcr(:, 2)*(Vc2'*Vc2));

R2wcrVc1 = h11 + h12;
R2wcrVc2 = h21 + h22;

%% CCA
[~, ~, Rx342Y] = canoncorr(X(:, [3, 4, 2]),Y);
R2x341Y = Rx342Y.^2;

%% LDA
y = [1, 1, 2, 2, 3, 3, 3]; % generate label for lda 
[~, Lambda, Sb, Sw] = f_lda(X(:, [3, 4, 2]), y);



