classdef (Abstract) Policy
% POLICY Generic class for policies.
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        daction % Size of the action drawn by the policy
        dparams % Size of the parameters of the policy
        theta   % Policy parameters
        no_bias % If true, the basis functions will not have the bias (no feature 1)
    end
    
    methods
        action = drawAction(obj, varargin);
        probability = evaluate(obj, varargin);
        obj = makeDeterministic(obj);
        obj = randomize(obj, varargin);
        obj = update(obj, theta);
        entropy = entropy(obj);
        
        %% To enable arrays equality, i.e., 
        % [pol1, pol2] == pol3 or [pol1, pol2] == [pol3, pol4]
        function areEq = eq(obj1, obj2)
            n1 = numel(obj1);
            n2 = numel(obj2);
            
            if n1 == n2
                areEq = isequal(obj1,obj2);
            elseif n1 == 1
                areEq = false(size(obj2));
                for i = 1 : n2
                    areEq(i) = isequal(obj2(i),obj1);
                end
            elseif n2 == 1
                areEq = false(size(obj1));
                for i = 1 : n1
                    areEq(i) = isequal(obj1(i),obj2);
                end
            else
                error('Matrix dimensions must agree.')
            end
        end
        
        %% 
        function phi = get_basis(obj, States)
            if obj.no_bias
                phi = obj.basis(States);
            else % Add the constant feature 1 (bias) to the basis functions
                phi = [ones(1,size(States,2)); obj.basis(States)];
            end
        end
    end
    
end
