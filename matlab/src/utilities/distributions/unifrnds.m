function r = unifrnds(a, b, n)
% UNIFRNDS Draws N random values from an uniform distribution in the 
% simplex of [A,B].
%
%    INPUT
%     - a : lower bound, vector of length D
%     - b : upper bound, vector of length D
%     - n : number of samples
%
%    OUTPUT
%     - r : random values, matrix of size [D x N]
%
% =========================================================================
% More information about sampling from the simplex:
% http://math.stackexchange.com/questions/502583/uniform-sampling-of-points-on-a-simplex

if ~iscolumn(a), a = a'; end
if ~iscolumn(b), b = b'; end
assert(min(a <= b) == 1, 'Bounds are not consistent.')

dim = length(a);

real_lo = a; % shift if lower bound is not 0
b = b - a;
a = a - a;
rnd = rand(dim, n);
rnd = -log(rnd);
r = rnd;
tot = sum(rnd);
tot = tot - log(rand(1, n));
r = bsxfun(@plus, a, bsxfun(@times, (b - a), r));
r = bsxfun(@times, r, 1 ./ tot);
r = bsxfun(@plus, r, real_lo); % shift back
