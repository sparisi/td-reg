function [J, ds] = show_simulation(mdp, policy, steps, pausetime, render)
% SHOW_SIMULATION Runs an episode and shows what happened during its 
% execution with an animation.
%
%    INPUT
%     - mdp       : the MDP to be seen
%     - policy    : the low level policy
%     - steps     : steps of the episode
%     - pausetime : time between animation frames
%     - render    : (optional) to render pixels generated from the MDP
%
%    OUTPUT
%     - J         : the total return of the episode
%     - ds        : the episode dataset

mdp.closeplot
[ds, J] = collect_samples(mdp, 1, steps, policy);

if nargin < 5, render = 0; end
if nargin < 4 || isempty(pausetime), pausetime = 0.01; end

if ~render
    mdp.plotepisode(ds, pausetime)
else
    [pixels, clims, cmap] = mdp.render([ds.s(:,1), ds.nexts]);
    fig = findobj('type','figure','name','Pixels Animation');
    if isempty(fig), fig = figure(); fig.Name = 'Pixels Animation'; end
    for i = 1 : size(pixels,3)
        clf, imagesc(pixels(:,:,i));
        fig.CurrentAxes.CLim = clims;
        colormap(cmap);
        drawnow limitrate
        if i ~= size(pixels,3), title(strrep(mat2str(ds.r(:,i)'), ' ', ', ')), end
        pause(pausetime)
    end
end
