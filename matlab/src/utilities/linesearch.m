function [success, x_new, n_back] = linesearch(f, x, fullstep, expected_improve_rate, g, eps)
% Backtracking linesearch for a constrained optimization problem
%
% max   f(x)
% s.t.  g(x) < eps
%
% FULLSTEP is the update direction.
% EXPECTED_IMPROVE_RATE is the slope df/dx at the initial point.

max_backtracks = 10;
accept_ratio = 0.1;
success = false;

fval = f(x);
n_back = 0;
for stepfrac = 0.5.^(0:max_backtracks-1)
    x_new = x + stepfrac * fullstep;
    try
        fval_new = f(x_new);
        if g(x_new) > eps
            fval_new = -inf;
        end
    catch
        fval_new = -inf;
    end
    actual_improve = fval_new - fval;
    expected_improve = expected_improve_rate * stepfrac;
    ratio = actual_improve / expected_improve;
    if ratio > accept_ratio && actual_improve > 0
        success = true;
        return
    end
    n_back = n_back + 1;
end
x_new = x;