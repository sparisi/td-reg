function samples = multimvnrnd(varargin)
% MULTIMVNRND Draws different samples from many Gaussian distribution, each
% one identified by its mean and the Cholesky decomposition of its
% covariance.
%
%    INPUT (for constant Gaussians)
%     - U       : [D x D x N] matrix of the Cholesky decompositions
%                 (Cov_i = U_i'*U_i)
%     - mu      : [D x N] matrix of the means
%
%    INPUT (for linear Gaussians)
%     - U       : [D x D x N] matrix of the Cholesky decompositions
%                 (Cov_i = U_i'*U_i)
%     - A       : [D x M x N] matrix of the linear components of the means
%     - phi     : [M x N] matrix of the features
%
%    OUTPUT
%     - samples : [D x N] matrix


if nargin == 2 % constant Gaussian: N(mu,U'U)
    U = varargin{1};
    mu = varargin{2};
    
    [d, n] = size(mu);
    r = randn(d,n);
    
    samples = zeros(d,n);
    for i = 1 : n
        samples(:,i) = U(:,:,i)' * r(:,i) + mu(:,i);
    end
elseif nargin == 3 % linear Gaussian: N(A*phi,U'U)
    U = varargin{1};
    A = varargin{2};
    phi = varargin{3};
    
    [~, d, n] = size(U);
    r = randn(d,n);
    
    samples = zeros(d,n);
    for i = 1 : n
        samples(:,i) = U(:,:,i)' * r(:,i) + A(:,:,i) * phi(:,i);
    end
else
    error('Wrong number of arguments.')
end
