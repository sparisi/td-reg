function updateplot(name, newX, newY, cursorOn)
% UPDATEPLOT Updates 2d plots with new points.
%
%    INPUT
%     - name     : figure name
%     - newX     : cell array of new single X-coord (one element per plot)
%     - newY     : cell array of new single Y-coord (one element per plot)
%     - cursorOn : (optional) any to show the coordinates of the new points
%
% =========================================================================
% EXAMPLE
% for i = 1 : 10, updateplot('Test',{i,i},{rand,rand*10}, 1), pause(0.5), end

% Look for a figure with the specified name
fig = findobj('type','figure','name',name);

if ~iscell(newX), newX = num2cell(newX); end
if ~iscell(newY), newY = num2cell(newY); end

% Plots share the same x coordinates
if numel(newX) == 1, newX = repmat(newX,1,numel(newY)); end

% If the figure does not exist, create it and plot the first points
if isempty(fig)
    fig = figure();
    fig.Name = name;
    
    % Get the figure's datacursor mode and activate it
    cursorMode = datacursormode(fig);
    set(cursorMode, 'enable', 'on');
    
    hold all
    for i = 1 : numel(newX)
        dataObjs = plot(newX{i}, newY{i});
        if nargin == 4
            set(dataObjs, 'UserData', cursorMode.createDatatip(dataObjs));
        end
    end
    
    hold off
    title(name)
    legend show
    return
end

% Find plots in the figure
dataObjs = findobj(fig,'Type','line');
X = get(dataObjs, 'XData')';
Y = get(dataObjs, 'YData')';
if ~iscell(X), X = {X}; Y = {Y}; end % If there is only one plot

% Append new points
for i = 1 : numel(dataObjs)
    X{i}(end+1) = newX{i};
    Y{i}(end+1) = newY{i};
    set(dataObjs(i), 'XData', X{i});
    set(dataObjs(i), 'YData', Y{i});
    
    if nargin == 4
        % Update annotation position
        set(get(dataObjs(i), 'UserData'), 'Position', [newX{i}, newY{i} 0])
    end
end

drawnow limitrate
