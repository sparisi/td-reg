function bfs = squared_bfs(s,a)
% Basis function composed as
%
%   [ 1
%     s
%     s^2
%     a
%     a^2
%     sa ]
%
% s^2, a^2 and sa include all cross-products (e.g., s1*s2, s1*s3, ...)

[ds, n] = size(s);
da = size(a,1);

bfs = [basis_quadratic(ds,s)
    a];

bfs_aa = [];
bfs_sa = [];
for i = 1 : da
    for j = 1 : da
        if j >= i
            bfs_aa = [bfs_aa
                a(i,:) .* a(j,:)];
        end
        bfs_sa = [bfs_sa
            s(i,:) .* a(j,:)];
    end
end

bfs = [bfs; bfs_aa; bfs_sa];
