function bfs_da = cubic_bfs_da(s,a)

[ds, n] = size(s);
[da, n] = size(a);

x = [ones(1,n); s; a];
dim = nmultichoosek(ds+da+1,3);
bfs_da = zeros(dim,da,n);

for z = 1 : da
    count = 1;
    idx_a = z + 1 + ds;
    for i = 1 : 1+ds+da
        for j = i : 1+ds+da
            for k = j : 1+ds+da
                if k == idx_a
                    bfs_da(count,z,:) = bfs_da(count,z,:) + permute(x(i,:).*x(j,:), [1 3 2]);
                end
                if j == idx_a
                    bfs_da(count,z,:) = bfs_da(count,z,:) + permute(x(i,:).*x(k,:), [1 3 2]);
                end
                if i == idx_a
                    bfs_da(count,z,:) = bfs_da(count,z,:) + permute(x(j,:).*x(k,:), [1 3 2]);
                end
                count = count + 1;
            end
        end
    end
end
