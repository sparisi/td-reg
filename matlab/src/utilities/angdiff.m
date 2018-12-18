function diff = angdiff(a, b, mode)
% ANGDIFF Computes the distance between angles A and B. MODE determines if
% angles are in radians (RAD) or degrees (DEG).

switch mode
    case 'deg', diff = 180 - abs(abs(bsxfun(@minus,a,b)) - 180); 
    case 'rad', diff = pi - abs(abs(bsxfun(@minus,a,b)) - pi);
    otherwise, error('Unknown mode.')
end