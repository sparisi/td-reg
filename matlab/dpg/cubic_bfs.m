function bfs = cubic_bfs(s,a)

[ds, n] = size(s);
[da, n] = size(a);

x = [ones(1,n); s; a];
bfs = [];

for i = 1 : 1+ds+da
    for j = i : 1+ds+da
        for k = j : 1+ds+da
            bfs(end+1,:) = x(i,:).*x(j,:).*x(k,:);
        end
    end
end
