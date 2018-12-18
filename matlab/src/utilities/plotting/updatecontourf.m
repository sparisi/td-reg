function updatecontourf(name, X, Y, newZ, c)
% UPDATECONTOURF Updates filled contour plots without generating a new figure.
%
%    INPUT
%     - name : figure name
%     - X    : X as in surf
%     - Y    : Y as in surf
%     - newZ : Z as in surf
%     - c    : (optional) 1 to display colorbar

if nargin < 5, c = 0; end

% Look for a figure with the specified name
fig = findobj('type','figure','name',name);

% If the figure does not exist, create it and plot the surf
if isempty(fig)
    fig = figure();
    fig.Name = name;
    s = contourf(X,Y,newZ);
    title(name)
    xlabel x
    ylabel y
    if c, colorbar, end
    return
end

% Update Z values
dataObj = findobj(fig,'Type','Contour');
set(dataObj, 'ZData', newZ);

drawnow limitrate
