function r = myunidrnd(a,b,n)
% MYUNIDRND Draws N random values from a discrete uniform distribution in 
% [A,B].
%
%    INPUT
%     - a : lower bound, vector of length D
%     - b : upper bound, vector of length D
%     - n : number of samples
%
%    OUTPUT
%     - r : random values, matrix of size [D x N]

if ~iscolumn(a), a = a'; end
if ~iscolumn(b), b = b'; end
assert(min(a <= b) == 1, 'Bounds are not consistent.')

if nargin < 3, n = 1; end

d = length(a);
r = floor( bsxfun(@plus, a, bsxfun(@times, (b + 1 - a), rand(d,n))) );
