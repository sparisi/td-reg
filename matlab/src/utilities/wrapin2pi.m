function theta = wrapin2pi(theta)
% WRAPIN2PI Wraps angles in radians to [0 2*pi]

ispositive = (theta > 0);
theta = mod(theta, 2*pi);
theta((theta == 0) & ispositive) = 2*pi;
