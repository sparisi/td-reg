function y = cumsummove(x, idx)
% CUMSUMMOVE Perform a cumulative sum over index windows. The sum is 
% performed along the second dimension.
%
% =========================================================================
% EXAMPLE
% x = [1 2 3 4 5 6]; idx = [0 0 1 0 1 0]; --> y = [1 3 3 7 5 11]

cx = cumsum(x,2);
idx = idx==1;
id = zeros(size(x));
diff_values = x(:,idx) - cx(:,idx);
id(:,idx) = diff([zeros(size(diff_values,1),1) diff_values],1,2);
y = cx + cumsum(id,2);
