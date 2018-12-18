function [data, J] = collect_samples(mdp, episodes, maxsteps, policy, contexts)
% COLLECT_SAMPLES Simulates episodes and provides low level details, i.e., 
% it collects tuples (s,a,s',r) at each time step.
% If you want to collect just the return of many different policies, please 
% see COLLECT_EPISODES.
%
%    INPUT
%     - mdp      : the MDP to be solved
%     - episodes : number of episodes to run
%     - maxsteps : max number of steps per episode
%     - policy   : policy followed by the agent
%     - contexts : (optional) contexts of each episode
%
%    OUTPUT
%     - data     : struct with the following fields (one per episode)
%                   * s        : state
%                   * a        : action
%                   * nexts    : next state
%                   * r        : immediate reward
%                   * terminal : 1 if the state is terminal, 0 otherwise
%                   * t        : time index
%                   * length   : length of the episode
%     - J        : returns averaged over all the episodes

assert(numel(policy) == 1, ...
    ['This function supports only one policy as input. ' ...
    'For collecting samples with multiple policies, see COLLECT_EPISODES. ' ...
    'For evaluating multiple policies, see EVALUATE_POLICIES'])

% Initialize variables
state = mdp.initstate(episodes);
totrew = zeros(mdp.dreward,episodes);
step = 0;

% Allocate memory
ds.s = nan(mdp.dstate, episodes, 1);
ds.nexts = nan(mdp.dstate, episodes, 1);
ds.a = nan(mdp.daction, episodes, 1);
ds.r = nan(mdp.dreward, episodes, 1);
ds.terminal = nan(1, episodes, 1);
ds.t = nan(1, episodes, 1);

% Keep track of the states which did not terminate
ongoing = true(1,episodes);

% Save the last step per episode
endingstep = maxsteps*ones(1,episodes);

% Run the episodes until maxsteps or all episodes end
while ( (step < maxsteps) && sum(ongoing) > 0 )
    
    step = step + 1;
    running_states = state(:,ongoing);
    
    % Select action
    action = policy.drawAction(running_states);
    
    % Simulate one step of all running episodes at the same time
    if nargin < 5
        [nextstate, reward, terminal] = mdp.simulator(running_states, action);
    else
        [nextstate, reward, terminal] = mdp.simulator(running_states, action, contexts(:,ongoing));
    end
    
    % Update the total reward
    totrew(:,ongoing) = totrew(:,ongoing) + (mdp.gamma)^(step-1) .* reward;
    
    % Record sample
    ds.a(:,ongoing,step) = action;
    ds.r(:,ongoing,step) = reward;
    ds.s(:,ongoing,step) = running_states;
    ds.nexts(:,ongoing,step) = nextstate;
    ds.terminal(:,ongoing,step) = terminal;
    ds.t(:,ongoing,step) = step;
    
    % Continue
    idx = 1:episodes;
    idx = idx(ongoing);
    idx = idx(terminal);
    state(:,ongoing) = nextstate;
    ongoing(ongoing) = ~terminal;
    endingstep(idx) = step;
    
end

% This permutation speeds up the creation of the struct
ds.s = permute(ds.s, [1 3 2]);
ds.a = permute(ds.a, [1 3 2]);
ds.nexts = permute(ds.nexts, [1 3 2]);
ds.r = permute(ds.r, [1 3 2]);
ds.terminal = permute(ds.terminal, [1 3 2]);
ds.t = permute(ds.t, [1 3 2]);

% Convert dataset to struct to allow storage of episodes with different length
data = struct( ...
    's', num2cell(ds.s,[1 2]), ...
    'a', num2cell(ds.a,[1 2]), ...
    'r', num2cell(ds.r,[1 2]), ...
    'nexts', num2cell(ds.nexts,[1 2]), ...
    'length', num2cell(permute(endingstep,[3 1 2]),1), ...
    'terminal', num2cell(ds.terminal,[1 2]), ...
    't', num2cell(ds.t,[1 2]) ...
    );
data = squeeze(data);

% Remove allocated (but not run) steps
for i = find(endingstep < max(endingstep)) 
    data(i).s = data(i).s(:,1:endingstep(i));
    data(i).r = data(i).r(:,1:endingstep(i));
    data(i).a = data(i).a(:,1:endingstep(i));
    data(i).nexts = data(i).nexts(:,1:endingstep(i));
    data(i).length = endingstep(i);
    data(i).terminal = data(i).terminal(:,1:endingstep(i));
    data(i).t = data(i).t(:,1:endingstep(i));
end

% If we are in the average reward setting, then normalize the return
if mdp.isAveraged && mdp.gamma == 1, totrew = bsxfun(@times, totrew, 1./ endingstep); end

J = mean(totrew,2);

return
