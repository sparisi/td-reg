function M2 = ndlinspace(M1,M2,N)
% NDLINSPACE - Generalized linearly spaced matrix for multiple points
%
%   R = NDLINSPACE(M1,M2, N) returns a matrix of size [size(M1) N] holding
%   N equally spaced points between corresponding point of M1 and M2. If N
%   is smaller than 2, M2 is returned.  R = NDLINSPACE(M1,M2) uses N = 100.
%
%   For scalar inputs, NDLINSPACE(S1,S2,N) mimicks LINSPACE, returning
%   equally spaced points between the S1 and S2. Example:
%
%      ndlinspace(0, 2, 4)
%        % -> 0  0.67    1.33    2.00
%
%   For two M-by-1 column vectors, NDLINSPACE(V1,V2,N) returns a M-by-N
%   matrix in which the k-th row holds the N equally spaced numbers between
%   V1(k) and V2(k). Similary, for two 1-by-M row vectors, a N-by-M matrix
%   is returned. Example:
%
%      ndlinspace([0 ; 12 ; 10], [3 ;18 ; 11],4)
%        % ->    0      1.00    2.00    3.00
%        %      12.00  14.00   16.00   18.00
%        %      10.00  10.33   10.67   11.00
%
%   For two ND arrays of size P-by-Q-by-R-by-.. matrices, NDLINSPACE
%   (M1,M2,N) returns a P-by-Q-by-R-by-..-by-N matrix, holding N equally
%   space points between M1(i,j,k,..) and M2(i,j,k,..). Example:
%
%      M1 = reshape(1:2*3*4*5,[2 3 4 5]) ;
%      M2 = 10 * M1 ;
%      size(M1) ; % ans =  2     3     4     5
%      R = ndlinspace(M1,M2,5) ;
%      squeeze(R(2,1,4,3,:)).'
%         % ans =  68   221   374   527   680
%      % Compare with LINSPACE for scalars
%      linspace(M1(2,1,4,3),M2(2,1,4,3),5)
%         % ans =  68   221   374   527   680
%
%  In general, M1 and M2 have to have the same size. Scalar expansion is
%  applied when one of the inputs is a scalar. Example:
%      ndlinspace(0,[4 ; 8 ; -12],5) % expand M1
%        % ans =  0     1     2     3     4
%        %        0     2     4     6     8
%        %        0    -3    -6    -9   -12
%
%  See also LINSPACE, LOGSPACE, REPMAT, INTERP1, SQUEEZE

% for Matlab R13 and up
% version 1.1 (feb 2009)
% (c) Jos van der Geest
% email: jos@jasen.nl

% History
% 1.0 (feb 2009) - inspired by a submissions on the File Exchange (#22824)
% 1.1 (feb 2009) - some minor textual corrections

if nargin == 2
    N = 100 ; % default
end

% scalar expansion
if numel(M1)==1 && numel(M2) > 1
    M1 = repmat(M1,size(M2)) ;
elseif numel(M2)==1 && numel(M1) > 1
    M2 = repmat(M2,size(M1)) ;
end

sz = size(M1) ;

if ~isequal(size(M2), sz)
    error('The two matrices M1 and M2 should have the same size.') ;
end

if N < 2 || isempty(M2),
    return
elseif numel(M1) == 1,
    % simple case for linspace
    M2 = [M1+(0:N-2)*(M2-M1)/(floor(N)-1) M2];
else
    % This is the real ND case. For each pair (M1(i), M2(i)) we have to find
    % the Y-values of the points along the straight line between (1,M1(i))
    % and (N M2(i)). The x-values are specified by the number 1:N.
    % We use linear interpolation for this. Pass a 2-by-P vector to INTERP1
    % and let that function handle the calculations in one step. See the
    % help of INTERP1 for more details.
   
    M2 = interp1([1 N],[M1(:) M2(:)].',1:N).' ;

    % Take care of the proper output format
    if numel(sz)==2 
        % return proper format for scalar and vector input
        M2 = squeeze(M2) ;
        % for row vector inputs return a column oriented matrix
        if sz(1)==1
            M2 = M2.' ;
        end
    else
        % for any matrix with ND dimensions (ND>2) return a ND+1
        % dimensional matrix. The last dimension holds the interpolated
        % equally spaced points.
        M2 = reshape(M2, [sz N]) ;
    end
end
