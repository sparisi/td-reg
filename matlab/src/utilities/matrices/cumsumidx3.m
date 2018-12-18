function out = cumsumidx3(v, i)
% CUMSUMIDX Perform an indices-wise cumulative sum over a matrix. The sum
% is performed along the third dimension.
% 
% See also CUMSUMIDX

assert(length(i) <= size(v,3), ...
    'The number of indices is higher than the number of elements.')

c = cumsum(v,3);
r = c(:,:,i);
out = cat(3, c(:,:,i(1)), diff(r,[],3));
