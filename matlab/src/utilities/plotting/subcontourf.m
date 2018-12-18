function subcontourf(name, X, Y, Z, c)
% SUBCONTOUR Plots multiple contourf on the same figure using subplot.
%
%    INPUT
%     - name : figure name
%     - X    : [N x M] matrix or [1 x M] vector
%     - Y    : [N x M] matrix or [1 x N] vector
%     - Z    : [D x MN] matrix
%     - c    : (optional) 1 to display colorbar

if nargin < 5, c = 0; end

[nplots, ~] = size(Z);
if isvector(X)
    m = length(X);
    n = length(Y);
else
    [n, m] = size(X);
end
nrows = floor(sqrt(nplots));
ncols = ceil(nplots/nrows);

fig = findobj('type','figure','name',name);

if isempty(fig)
    figure('Name',name);
    for i = nplots : -1 : 1
        subplot(nrows,ncols,i,'align')
        contourf(X,Y,reshape(Z(i,:),n,m))
        if nplots > 1, title(num2str(i)), end
        if c, colorbar, end
    end
else
    axes = findobj(fig,'type','axes');
    for i = nplots : -1 : 1
        axes(i).Children.XData = X;
        axes(i).Children.YData = Y;
        axes(i).Children.ZData = reshape(Z(i,:),n,m);
    end
end

drawnow limitrate
