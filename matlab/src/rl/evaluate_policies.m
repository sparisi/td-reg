function J = evaluate_policies(mdp, episodes, maxsteps, policies, contexts)
% EVALUATE_POLICIES Evaluates a set of policies. For each policy, several 
% episodes are simulated.
% The function is very similar to COLLECT_SAMPLES, but it accepts many
% policies as input and it does not return the low level dataset.
%
%    INPUT
%     - mdp      : the MDP to be solved
%     - episodes : number of episodes per policy
%     - maxsteps : max number of steps per episode
%     - policy   : policies to be evaluated
%     - contexts : (optional) contexts of each episode
%
%    OUTPUT
%     - J        : returns of each policy

npolicy = numel(policies);
totepisodes = episodes * npolicy;

% Initialize variables
J = zeros(mdp.dreward,totepisodes);
step = 0;

% Initialize simulation
state = mdp.initstate(totepisodes);
action = zeros(mdp.daction,totepisodes);

% Keep track of the states which did not terminate
ongoing = true(1,totepisodes);

% Duplicate contexts for indexing
if nargin == 5, contexts = repmat(contexts,1,episodes); end

% Save the last step per episode
endingstep = maxsteps*ones(1,totepisodes);

% Run the episodes until maxsteps or all ends
while ( (step < maxsteps) && sum(ongoing) > 0 )

    step = step + 1;
    
    % Select action
    for i = 1 : npolicy
        idx = (i-1)*episodes+1 : (i-1)*episodes+episodes;
        doStates = state(:, idx(ongoing(idx)));
        if ~isempty(doStates)
            action(:,idx(ongoing(idx))) = policies(i).drawAction(state(:, idx(ongoing(idx))));
        end
    end

    % Simulate one step of all running episodes at the same time
    if nargin < 5
        [nextstate, reward, terminal] = mdp.simulator(state(:,ongoing), action(:,ongoing));
    else
        [nextstate, reward, terminal] = mdp.simulator(state(:,ongoing), action(:,ongoing), contexts(:,ongoing));
    end
    state(:,ongoing) = nextstate;
    
    % Update the total reward
    J(:,ongoing) = J(:,ongoing) + (mdp.gamma)^(step-1) .* reward;
    
    % Continue
    idx = 1:totepisodes;
    idx = idx(ongoing);
    idx = idx(terminal);
    endingstep(idx) = step;
    ongoing(ongoing) = ~terminal;
    
end

% If we are in the average reward setting, then normalize the return
if mdp.isAveraged && mdp.gamma == 1, J = bsxfun(@times, J, 1 ./ endingstep); end

J = permute( mean( reshape(J,[mdp.dreward episodes npolicy]), 2), [1 3 2] );

return
