classdef (Abstract) LQREnv < handle
% This class has closed form equations for the linear quadratic regulator (LQR).
% a = K*s    K < 0
% s' = A*s + B*a
% r = -x'*Q*x -a'*R*a     Q > 0, R > 0
    
    properties
        A
        B
        x0
        Q
        R
    end
    
    methods

        %% Closed form equations
        function P = riccati(obj, K)
            g = obj.gamma;
            A = obj.A;
            B = obj.B;
            R = obj.R;
            Q = obj.Q;
            num_obj = size(obj.Q,3);
            I = eye(size(A));
            P = zeros(size(obj.Q));
            
            for i = 1 : num_obj
                if isequal(A, B, I) && isdiag(Q(:,:,i)) && isdiag(R(:,:,i)) && isdiag(K)
                    P(:,:,i) = (Q(:,:,i) + K .* R(:,:,i) .* K) ./ (I - g * (I + 2 * K + K.^2));
                    P(:,:,i) = diag(diag(P(:,:,i)));
                else
                    tolerance = 0.0001;
                    converged = false;
                    iter = 0; maxiter = 500;
                    P(:,:,i) = I;
                    Pnew(:,:,i) = Q(:,:,i) + g*A'*P(:,:,i)*A + g*K'*B'*P(:,:,i)*A + g*A'*P(:,:,i)*B*K + g*K'*B'*P(:,:,i)*B*K + K'*R(:,:,i)*K;
                    while ~converged
                        P(:,:,i) = Pnew(:,:,i);
                        Pnew(:,:,i) = Q(:,:,i) + g*A'*P(:,:,i)*A + g*K'*B'*P(:,:,i)*A + g*A'*P(:,:,i)*B*K + g*K'*B'*P(:,:,i)*B*K + K'*R(:,:,i)*K;
                        converged = max(abs(P(:)-Pnew(:))) < tolerance;
                        iter = iter + 1;
                        if isnan(max(abs(P(:)-Pnew(:)))), error('LQR system unstable!'), end
                        if iter > maxiter, break, end % LQR unstable, but we want an approximate solution
                    end
                end
            end
        end
   
        function Vf = v_function(obj, K, Sigma, s)
            P = obj.riccati(K);
            g = obj.gamma;
            B = obj.B;
            R = obj.R;

            for i = 1 : size(obj.Q,3)
                if g == 1
                    Vf(i,:) = -sum(bsxfun(@times, s'*P(:,:,i), s'), 2)';
                else
                    Vf(i,:) = -sum(bsxfun(@times, s'*P(:,:,i), s'), 2)' - (1/(1-g)) * trace( Sigma * (R(:,:,i) + g*B'*P(:,:,i)*B) );
                end
            end
        end
        
        function Qmodel = q_model(obj, K, Sigma)
        % QMODEL is a struct with all matrices of the quadratic model of 
        % the true Q-function, namely Q0, Qss, Qaa, Qsa. The model is
        %    Q = Q0 + s'*Qss*s + a'*Qaa*a + s'*Qsa*a
        %
        % Additionally, it merges all these matrices into a bigger one
        %    Q = [1 s a]'*H*[1 s a]
            P = obj.riccati(K);
            g = obj.gamma;
            A = obj.A;
            B = obj.B;
            R = obj.R;
            Q = obj.Q;
            num_obj = size(obj.Q,3);
            
            for i = 1 : num_obj
                if g < 1
                    Qmodel(i).Qss = -Q(:,:,i) - g*A'*P(:,:,i)*A;
                    Qmodel(i).Qaa = -R(:,:,i) - g*B'*P(:,:,i)*B;
                    Qmodel(i).Qsa = -2*g*B'*P*A;
                    Qmodel(i).Q0  = -(g/(1-g))*trace( Sigma * (R(:,:,i) + g*B'*P(:,:,i)*B) );
                else
                    Qmodel(i).Qss = -Q(:,:,i) - A'*P(:,:,i)*A;
                    Qmodel(i).Qaa = -R(:,:,i) - B'*P(:,:,i)*B;
                    Qmodel(i).Qsa = -2*B'*P*A;
                    Qmodel(i).Q0  = trace( Sigma * (R(:,:,i) + B'*P(:,:,i)*B) );
                end
                Qmodel(i).H = [Qmodel(i).Qss   , Qmodel(i).Qsa/2
                               Qmodel(i).Qsa/2 , Qmodel(i).Qaa];
                Qmodel(i).H = blkdiag(Qmodel(i).Q0, Qmodel(i).H);
            end
        end
        
        function Qf = q_function(obj, K, Sigma, s, a)
            P = obj.riccati(K);
            g = obj.gamma;
            A = obj.A;
            B = obj.B;
            R = obj.R;
            Q = obj.Q;
            num_obj = size(obj.Q,3);
            
            tmp = (A*s + B*a);
            
            for i = 1 : num_obj
                if g == 1
                    Qf(i,:) = -sum(bsxfun(@times, s'*Q(:,:,i), s'), 2)' ...
                        -sum(bsxfun(@times, a'*R(:,:,i), a'), 2)' ...
                        -g*sum(bsxfun(@times, tmp'*P(:,:,i), tmp'), 2)' ...
                        +trace( Sigma * (R(:,:,i) + g*B'*P(:,:,i)*B) );
                else
                    Qf(i,:) = -sum(bsxfun(@times, s'*Q(:,:,i), s'), 2)' ...
                        -sum(bsxfun(@times, a'*R(:,:,i), a'), 2)' ...
                        -g*sum(bsxfun(@times, tmp'*P(:,:,i), tmp'), 2)' ...
                        -(g/(1-g))*trace( Sigma * (R(:,:,i) + g*B'*P(:,:,i)*B) );
                end
            end
        end
        
        function Af = a_function(obj, K, Sigma, s, a)
            Af = obj.q_function(K,Sigma,s,a) - obj.v_function(K,Sigma,s);
        end
        
        function J = avg_return(obj, K, Sigma)
            P = obj.riccati(K);
            num_obj = size(obj.Q,3);
            J = zeros(num_obj,1);
            B = obj.B;
            R = obj.R;
            g = obj.gamma;
            x0 = obj.x0;

            for i = 1 : num_obj
                if g == 1
                    J(i,:) = - trace(Sigma*(R(:,:,i)+B'*P(:,:,i)*B));
                else
                    % If the init state is fixed, use the first equation. 
                    % If it is drawn from a unif distrib in [-x0,x0], use the second.

%                     J(i,:) = - (x0'*P(:,:,i)*x0 + (1/(1-g))*trace(Sigma*(R(:,:,i)+g*B'*P(:,:,i)*B)));

                    mu_s = (x0 - x0)/2;
                    Sigma_s = diag(x0 + x0).^2 / 12;
                    J(i,:) = - (trace(Sigma_s*P(:,:,i)) + mu_s'*P(:,:,i)*mu_s + (1/(1-g))*trace(Sigma*(R(:,:,i)+g*B'*P(:,:,i)*B)));
                end
            end
        end
        
        function [K, P] = opt(obj)
            for i = 1 : size(obj.Q,3)
                [x,l,g] = dare(sqrt(obj.gamma)*obj.A,sqrt(obj.gamma)*obj.B,obj.Q(:,:,i),obj.R(:,:,i));
                K(:,:,i) = -g;
                P(:,:,i) = x;
            end
        end
        
    end
end
