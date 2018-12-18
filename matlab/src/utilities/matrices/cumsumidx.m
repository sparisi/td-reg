function y = cumsumidx(x, idx)
% CUMSUMIDX Perform an indices-wise cumulative sum over a matrix. The sum
% is performed along the second dimension.
% 
% =========================================================================
% EXAMPLE
% x = [1:10; 11:20]; idx = [3 6 10];
% In this case we want the cumulative sum of x(:,1:3), x(:,4:6), x(:,7:10).
% Therefore, the result is [6 15 34; 36 45 74].

assert(length(idx) <= size(x,2), ...
    'The number of indices is higher than the number of elements.')

c = cumsum(x,2);
r = c(:,idx);
y = [c(:,idx(1)) diff(r,[],2)];
