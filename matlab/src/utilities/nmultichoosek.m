function combs = nmultichoosek(values, k)
% NMULTICHOOSEK Like nchoosek, but with repetitions. The VALUES for which
% nchoosek is performed are columns. If VALUES is a matrix, nchoosek is
% performed for each column and COMBS is a matrix as well.

[d, ncombs] = size(values);
if d == 1
    d = values;
    combs = nchoosek(d+k-1, k);
else
    if k == 2
        combs = nchoose2(1:d+k-1);
        combs(:,2) = combs(:,2) - 1;
    else
        combs = bsxfun(@minus, nchoosek(1:d+k-1,k), 0:k-1);
    end
    
    combs = reshape(values(combs,:), [], k, ncombs);
end