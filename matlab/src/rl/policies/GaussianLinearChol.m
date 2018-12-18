classdef GaussianLinearChol < GaussianLinear
% GAUSSIANLINEARCHOL Gaussian distribution with linear mean and constant 
% covariance: N(A*phi,S).
% Parameters: mean A and Cholesky decomposition U, with S = U'U.
%
% U is stored row-wise, e.g:
% U = [u1 u2 u3; 
%      0  u4 u5; 
%      0  0  u6] -> (u1 u2 u3 u4 u5 u6)
    
    methods
        
        %% Constructor
        function obj = GaussianLinearChol(basis, dim, initA, initSigma, no_bias)
            if nargin == 4, no_bias = false; end
            obj.no_bias = no_bias; 
            assert(isscalar(dim) && ...
            	size(initA,2) == basis()+1*~no_bias && ...
            	size(initA,1) == dim && ...
                size(initSigma,1) == dim && ...
            	size(initSigma,2) == dim, ...
                'Dimensions are not consistent.')
            [initCholU, p] = chol(initSigma);
            assert(p == 0, 'Covariance must be positive definite.')
            
            obj.daction = dim;
            obj.basis = basis;
            init_tri = initCholU';
            init_tri = init_tri(tril(true(dim), 0)).';
            obj.theta = [initA(:); init_tri'];
            obj.dparams = length(obj.theta);
            obj = obj.update(obj.theta);
        end
        
        %% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, state, action)
            nsamples = size(state,2);
            phi = obj.get_basis(state);
            A = obj.A;
            mu = A*phi;
            cholU = obj.U;
            invU = inv(cholU);
            invUT = invU';
            invSigma = invU * invU';
            diff = action-mu;
            dlogpdt_A = mtimescolumn(invSigma*diff, phi);

            idx = tril(true(obj.daction));
            nelements = sum(sum(idx));

            tmp = bsxfun(@plus,-invUT(:),mtimescolumn(invUT*diff, invSigma*diff));
            tmp = reshape(tmp,obj.daction,obj.daction,nsamples);
            tmp = permute(tmp,[2 1 3]);
            idx = repmat(idx,1,1,nsamples);
            dlogpdt_cholU = tmp(idx);
            dlogpdt_cholU = reshape(dlogpdt_cholU,nelements,nsamples);

            dlogpdt = [dlogpdt_A; dlogpdt_cholU];
        end

        %% WML
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            assert(min(weights)>=0, 'Weights cannot be negative.')
            assert(size(Phi,1) == obj.basis()+1*~obj.no_bias)
            weights = weights / sum(weights);
            PhiW = bsxfun(@times,Phi,weights);
            tmp = PhiW * Phi';
            if rank(tmp) == size(Phi,1)
                A = tmp \ PhiW * Action';
            else
                A = pinv(tmp) * PhiW * Action';
            end
            A = A';
            
            diff = bsxfun(@minus, Action, A*Phi);
            Sigma = sum( bsxfun( @times, ...
                permute(bsxfun(@times,diff, weights),[1 3 2]), ...
                permute(diff,[3 1 2]) ), 3);
            
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            Sigma = Sigma / Z;
            Sigma = nearestSPD(Sigma);
            cholU = chol(Sigma);
            tri = cholU';
            tri = tri(tril(true(obj.daction), 0)).';
            obj = obj.update([A(:); tri']);
        end

        %% Update
        function obj = update(obj, varargin)
            if nargin == 2 % Update by params
                theta = varargin{1};
                obj.theta(1:length(theta)) = theta;
                n = length(obj.theta) - sum(1:obj.daction);
                A = vec2mat(obj.theta(1:n),obj.daction);
                indices = tril(ones(obj.daction));
                cholU = indices;
                cholU(indices == 1) = obj.theta(n+1:end);
                cholU = cholU';
                obj.A = A;
                obj.U = cholU;
                obj.Sigma = cholU'*cholU;
            elseif nargin == 3 % Update by mean and covariance
                obj.A = varargin{1};
                obj.Sigma = varargin{2};
                [U, p] = chol(varargin{2});
                assert(p == 0, 'Covariance must be positive definite.')
                obj.U = U;
                U = U';
                U = U(tril(true(obj.daction), 0)).';
                obj.theta = [obj.A(:); U'];
            else
                error('Wrong number of input arguments.')
            end
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            n = numel(obj.A);
            obj.theta(n+1:end) = 0;
            obj = obj.update(obj.theta);
        end
        
        function obj = randomize(obj, factor)
            n = numel(obj.A);
            obj.theta(n+1:end) = obj.theta(n+1:end) * factor;
            obj = obj.update(obj.theta);
        end
        
    end
    
end
