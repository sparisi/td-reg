classdef DoubleLink < NLinkEnv

    %% Properties
    properties
        % Environment variables
        lengths = [1 1];
        masses = [1 1];
        inertias = [1 1] .* ([1 1].^2 + 0.0001) ./ 3.0;
        friction_coeff = [2.5 2.5]'; % Viscous friction coefficients
        g = 9.81; % If 0, the problem becames a simpler planar reaching task
        dt = 0.02;
        
        x_des = [0 2]'; % Goal in task space
        q_des = [pi/2 0]'; % Goal in joint space
        
        % MDP variables
        dstate = 4;
        daction = 2;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;

        % Bounds : state = [q1 qd1 q2 qd2]
        stateLB = [-pi, -50, -pi, -50]';
        stateUB = [pi, 50, pi, 50]';
        actionLB = [-10, -10]';
        actionUB = [10, 10]';
        rewardLB = -inf;
        rewardUB = 0;
    end
    
    methods

        %% Dynamics
        function nextstate = dynamics(obj, state, action)
            [gravity, coriolis, invM, friction] = obj.getDynamicsMatrices(state);
            qdd = mtimes32(invM, action - coriolis - gravity - friction);
            qd = state(2:2:end,:) + obj.dt * qdd;
            q = state(1:2:end,:) + obj.dt * qd;
            nextstate = [q(1,:); qd(1,:); q(2,:); qd(2,:)];
        end
        
        function [gravity, coriolis, invM, friction] = getDynamicsMatrices(obj, state)
            inertia1 = obj.inertias(1);
            inertia2 = obj.inertias(1);
            m1 = obj.masses(1);
            m2 = obj.masses(2);
            l1 = obj.lengths(1);
            l2 = obj.lengths(2);
            lg1 = l1 / 2;
            lg2 = l2 / 2;
            q1 = state(1,:);
            q2 = state(3,:);
            q1d = state(2,:);
            q2d = state(4,:);
%             s1 = sin(q1);
            c1 = cos(q1);
            s2 = sin(q2);
            c2 = cos(q2);
%             s12 = sin(q1 + q2);
            c12 = cos(q1 + q2);
            
            M11 = m1 * lg1^2 + inertia1 + m2 * (l1^2 + lg2^2 + 2 * l1 * lg2 * c2) + inertia2;
            M12 = m2 * (lg2^2 + l1 * lg2 * c2) + inertia2;
            M21 = M12;
            M22 = repmat(m2 * lg2^2 + inertia2, 1, size(state,2));
            invdetM = 1 ./ (M11 .* M22 - M12 .* M21);
            invM(1,1,:) = M22;
            invM(1,2,:) = -M21;
            invM(2,1,:) = -M12;
            invM(2,2,:) = M11;
            invM = bsxfun(@times, invM, permute(invdetM, [3 1 2]));

            gravity = [m1 * obj.g * lg1 * c1 + m2 * obj.g * (l1 * c1 + lg2 * c12)
                m2 * obj.g * lg2 * c12];
            coriolis = [-m2 * l1 * lg2 * s2 .* (2 .* q1d .* q2d + q2d.^2)
                m2 * l1 * lg2 * s2 .* q1d.^2];
            friction = [obj.friction_coeff(1) * q1d
                obj.friction_coeff(2) * q2d];
        end

        %% Kinematics
        function X = getJointsInTaskSpace(obj, state)
        % X = [ x1 y1 x1d y1d x2 y2 x2d y2d ]
            q1 = state(1,:);
            qd1 = state(2,:);
            q2 = state(3,:);
            qd2 = state(4,:);
            xy1 = obj.lengths(1) .* [cos(q1); sin(q1)];
            xy2 = xy1 + obj.lengths(2) .* [cos(q2+q1); sin(q2+q1)];
            
            xy1d = obj.lengths(1) .* [qd1.*cos(q1); -qd1.*sin(q1)];
            xy2d = xy1d + obj.lengths(2) .* [(qd1+qd2).*cos(q1+q2); -(qd1+qd2).*sin(q1+q2)];
            
            X = [xy1; xy1d; xy2; xy2d];
        end
        
    end
     
end