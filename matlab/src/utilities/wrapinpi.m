function theta = wrapinpi(theta)
% WRAPINPI Wraps angles in radians to [-pi pi]

isout = (theta < -pi) | (pi < theta);
theta(isout) = wrapin2pi(theta(isout) + pi) - pi;
