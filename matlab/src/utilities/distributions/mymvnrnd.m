function r = mymvnrnd(mu, sigma, n)
% MYMVNRND Draws N random values from a multivariate normal distribution of 
% mean MU and covariance SIGMA. MU can be either a single column of length
% D or a [D x N] matrix representing a set of means. In the former case, N 
% values R are drawn from a distribution with the same mean, in the latter 
% one value is drawn from N different distributions, one for each mean. In 
% both cases, the output R will always be a [D x N] matrix.

[d,nmeans] = size(mu);
if nargin == 2, n = nmeans; end
assert(iscolumn(mu) || nmeans == n, ...
    'The mean must be either a column or a matrix of N columns.')

u = chol(sigma);
r = bsxfun(@plus, u'*randn(d,n), mu);
