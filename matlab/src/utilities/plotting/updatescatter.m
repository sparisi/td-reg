function updatescatter(name, X, Y, Z, value, size)
% UPDATESCATTER Updates scatter plots without generating a new figure.
% The call is scatter(X,Y,SIZE,VALUE) or scatter3(X,Y,Z,SIZE,VALUE) if Z is 
% not empty.
% By default, SIZE = 1.

if nargin < 6, size = 1; end

% Look for a figure with the specified name
fig = findobj('type','figure','name',name);

% If the figure does not exist, create it and plot the surf
if isempty(fig)
    fig = figure();
    fig.Name = name;
    if isempty(Z), scatter(X,Y,size,value),
    else, scatter3(X,Y,Z,1,value), end
    title(name)
    xlabel x
    ylabel y
    return
end

% Update Z values
dataObj = findobj(fig,'Type','Scatter');
set(dataObj, 'XData', X);
set(dataObj, 'YData', Y);
set(dataObj, 'ZData', Z);
set(dataObj, 'CData', value);

drawnow limitrate
