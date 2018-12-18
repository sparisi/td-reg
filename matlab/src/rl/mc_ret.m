function R = mc_ret(data, gamma)
% Computes Monte-Carlo estimates of the return from potentially off-policy data.
% R_t = sum_(h=t)^T gamma^(h-t)*r_h
% Data have to be ordered by episode: data.r must have first all samples 
% from the first episode, then all samples from the second, and so on.
% So you cannot use samples collected with COLLECT_SAMPLES2.

r = [data.r];
t = [data.t];
t(end+1) = 1;
R = zeros(size(r));

for k = size(R,2) : -1 : 1
    if t(k+1) == 1 % Next state is a new episode init state
        R(k) = r(k);
    else
        R(k) = r(k) + gamma * R(k+1);
    end
end
