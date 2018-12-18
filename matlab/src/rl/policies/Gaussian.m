classdef (Abstract) Gaussian < Policy
% GAUSSIAN Generic class for Gaussian distributions.
    
    methods
        
        function S = entropy(obj, varargin)
        % Differential entropy, can be negative
            S = 0.5 * ...
                ( obj.daction*log(2*pi*exp(1)) + 2*sum(log(diag(obj.U))) );
        end

    end

end
