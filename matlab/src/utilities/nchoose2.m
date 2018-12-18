function Y = nchoose2(X)
% NCHOOSE2 - all combinations of two elements
%   Y = NCHOOSE2(X) returns all combinations of two elements of the array X.
%   It is the fast, vectorized version of NCHOOSEK(X,2).  X can be any type
%   of array.
%
%   Example:
%    nchoose2([10 20 30 40])
%     % -> 10    20
%     %    10    30
%     %    10    40
%     %    20    30
%     %    20    40
%     %    30    40
%
%    nchoose2({'a','b','c','d','e'})
%     % -> 'a'  'b'
%     %    'a'  'c'
%     %      ...
%     %    'c'  'e'
%     %    'd'  'e'
%
%   See also NCHOOSEK, PERMS
%            COMBN, NCHOOSE, ALLCOMB (on the File Exchange)

% for Matlab R13+
% version 2.1 (jun 2008)
% (c) Jos van der Geest
% email: jos@jasen.nl

% History
% 1.0, sep 2007 - created, for faster solution of nchoosek(x,2)
% 2.0, may 2008 - inspired to put on the FEX, by submission #20110 by S.
%                 Scaringi, and review by John D'Errico
%               - optimized engine, added extensive help and comments
% 2.1, jun 2008 - catch error when X has less than two elements
%                 (error pointed out by Urs Schwarz)

N = numel(X) ;
if N<2
    warning('nchoose2:InputTooSmall','Input has less than two elements') ;
    Y = [] ;    
elseif N==2,
    Y = X(:).' ; % output is a row vector
else 
% by creating an (N*(N-1)/2)-by-2 index matrix (where N is the number of
% elements of X), the output can be retrieved directly. This index matrix
% equals nchoosek(1:numel(X),2)
                                    % Example for N = 4 ->
    V  = N-1:-1:2 ;               %  V : 3 2
    ri = cumsum([1 V],2) ;          % ri : 1 4 6

% putting a row matrix into a column works because ind does not exist yet.
% it will also fill the first column with zeros.
    ind(ri,2) = [0 -V] + 1 ;        % ind -> c1: 0 0 0  0 0  0]
                                    %        c2: 1 0 0 -2 0 -1]
    ind(ri,1) = 1 ;                 % ind -> c1: 1 0 0  1 0  1]
                                    %        c2: 1 0 0 -2 0 -1]
    ind(:,2) = ind(:,2) + 1 ;       % ind -> c1: 1 0 0  1 0  1]
                                    %        c2: 2 1 1 -1 1  0]
    ind = cumsum(ind,1) ;           % ind -> c1: 1 1 1 2 2 3]
                                    %        c2: 2 3 4 3 4 4]
    Y = X(ind) ;                    % index into X
end

%   Notes:
%   - NCHOOSE2(X) is much faster than NCHOOSEK(X,2). It is also faster than
%     another solution ("nCtwo", FEX # 20110, May 28th 2008, Simone
%     Scaringi), especially for smaller arrays. It is also more memory
%     efficient than a solution based on a suggestion by John D'Errico in
%     his review of FEX #20110:
%        [I,J] = find(tril(ones(numel(x)),-1));
%        y = x([J(:) I(:)]);
%     The latter solution is a little faster for shorter vectors, but
%     slower and memory consuming for larger vectors. Moreover, it may
%     require a call to sortrows to get the same order as nchoosek.
%   - specifying the dimension for cumsum is slightly faster