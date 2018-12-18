function subsurf(name, X, Y, Z, addcolorbar)
% SUBSURF Plots multiple surf on the same figure using subplot.
%
%    INPUT
%     - name : figure name
%     - X    : [1 x M] vector
%     - Y    : [1 x N] vector
%     - Z    : [D x MN] matrix

[nplots, ~] = size(Z);
m = length(X);
n = length(Y);
nrows = floor(sqrt(nplots));
ncols = ceil(nplots/nrows);

fig = findobj('type','figure','name',name);

if isempty(fig)
    if nargin == 4, addcolorbar = 0; end
    figure('Name',name);
    for i = nplots : -1 : 1
        subplot(nrows,ncols,i,'align')
        surf(X,Y,reshape(Z(i,:),n,m))
        if nplots > 1, title(num2str(i)), end
        if addcolorbar, colorbar, end
    end
else
    axes = findobj(fig,'type','axes');
    for i = nplots : -1 : 1
        axes(i).Children.CData = reshape(Z(i,:),n,m);
        axes(i).Children.ZData = reshape(Z(i,:),n,m);
    end
end

drawnow limitrate
