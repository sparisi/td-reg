function r = mymnrnd(p, n)
% MYMNRND Draws N samples from a multimonial distribution of probabilities 
% P. Either P is a vector of size M or a matrix of size [M x N], where M is
% the number of variables of the distribution. In the former case, N 
% samples are drawn from the same distribution P. In the latter, N samples
% are drawn from each of the P distributions (one for each column). In both 
% cases, the random values R will be stored in a row vector.
%
% =========================================================================
% EXAMPLE
% p = [0.1 0.2 0.3 0.4];
% n = 10;
% mymnrnd(p, n) will draw ten integers between [1, 4]

if isvector(p)
    if ~iscolumn(p), p = p'; end
elseif size(p,2) ~= n
    error('p has wrong dimensions.');
end

m = size(p,1);
idx0 = sum(p,1) < 1e-20;
p(:,~idx0) = bsxfun(@times, p(:,~idx0), 1 ./ sum(p(:,~idx0),1)); % Ensure that p represents a distribution
p(:,idx0) = 1/m;
F = cumsum(p,1);
temp = bsxfun(@ge,F,rand(1,n));
r = m - sum(temp,1) + 1;
