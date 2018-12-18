function dist = avg_pairwise_dist(x)
% AVG_PAIRWISE_DIST Computes the average pairwise distance between samples.
%
%    INPUT
%     - x     [D-by-N] matrix, where D is the dimensionality of the data
%             and N is the number of samples
%
%    OUTPUT
%     - dist : [D-by-1] vector representing the state-wise distance.


dist = 0;
n = size(x,2);
for i = 1 : n
    dist = dist + mean(abs(bsxfun(@minus, x, x(:,i))),2);
end
dist = dist / (n-1);
