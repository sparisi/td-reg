function [f, df, ddf] = mse_linear(w, X, T)
% Mean squared error for linear functions
% min_w ||Y-T||^2,   Y = w'X
%
% X and T must be [D x N] matrices, where D is the dimensionality of
% the data and N is the number of samples.

Y = w'*X;
E = Y - T; % T are the targets
f = 0.5*mean(mean(E.^2)); % Function, MSE
df = X*mean(E,1)'/size(T,2); % Gradient
ddf = X*X'/size(T,2); % Hessian
