function bfs_da = squared_bfs_da(s,a)
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

aa_da = [];
sa_da = [];
for i = 1 : da
    tmp = zeros(da-i+1,da,n);
    for j = 1 : da
        if j >= i
            tmp(j-i+1,j,:) = a(i,:);
            tmp(j-i+1,i,:) = tmp(j-i+1,i,:) + permute(a(j,:), [1 3 2]);
        end
        tmp2 = zeros(1,da,n);
        tmp2(1,j,:) = s(i,:);
        sa_da = vertcat(sa_da,tmp2);
    end
    aa_da = vertcat(aa_da,tmp);
end

one_da = zeros(1,da,n);
s_da   = zeros(ds,da,n);
a_da   = repmat(eye(da),1,1,n);
ss_da  = zeros(nchoosek(ds+1,2),da,n);

bfs_da = vertcat(one_da, s_da, ss_da, a_da, aa_da, sa_da);
