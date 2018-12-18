classdef ADAM < handle
    
    properties
        alpha = 1e-3;
        beta1 = 0.9;
        beta2 = 0.999;
        epsilon = 1e-8;
        lambda = 0;
        t = 0;
        m = 0;
        v = 0;
    end
    
    methods
        
        function obj = ADAM(dim)
            obj.m = zeros(1,dim);
            obj.v = zeros(1,dim);
        end
        
        function x = step(obj, x, dx)
            obj.t = obj.t + 1;
            obj.m = obj.beta1 * obj.m + (1 - obj.beta1) * dx; % Update biased 1st moment estimate
            obj.v = obj.beta2 * obj.v + (1 - obj.beta2) * dx.^2; % Update biased 2nd raw moment estimate
            mhat = obj.m / (1 - obj.beta1^obj.t); % Compute bias-corrected 1st moment estimate
            vhat = obj.v / (1 - obj.beta2^obj.t); % Compute bias-corrected 2nd raw moment estimate
            x = x - obj.alpha * mhat ./ (sqrt(vhat) + obj.epsilon) ...
                - obj.alpha * obj.lambda * x; % Update with l2 weight decay
        end
        
    end
    
end