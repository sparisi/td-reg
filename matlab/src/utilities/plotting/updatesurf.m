function updatesurf(name, X, Y, newZ, opacity)
% UPDATESURF Updates surf plots without generating a new figure.
%
%    INPUT
%     - name    : figure name
%     - X       : X as in surf
%     - Y       : Y as in surf
%     - newZ    : Z as in surf
%     - opacity : (optional, default 1)

% Look for a figure with the specified name
fig = findobj('type','figure','name',name);

if nargin < 5, opacity = 1; end

% If the figure does not exist, create it and plot the surf
if isempty(fig)
    fig = figure();
    fig.Name = name;
    s = surf(X,Y,newZ);
    s.FaceAlpha = opacity;
    title(name)
    xlabel x
    ylabel y
    return
end

% Update Z values
dataObj = findobj(fig,'Type','Surface');
set(dataObj, 'CData', newZ);
set(dataObj, 'ZData', newZ);

drawnow limitrate
