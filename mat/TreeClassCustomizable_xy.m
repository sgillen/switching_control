classdef TreeClassCustomizable_xy < rl.env.MATLABEnvironment
    %TREECLASS: Template for defining custom environment in MATLAB.    
    
    %% Properties 
    properties
       g = -5 % The constant acceleration in y. 
       yinit = 2; % yvalue to inizialize y at 
       dt = .1; 
       tol = 0.1; % tolerance for accepting that we are in a deadzone
       N = 5; % how many steps i n an episode %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       deadzones = [3,7,0]; % where does the tree live?
       
       n_obs = 1; % number of obstacles in the env
       w_obs = 2; % width of the obstacles
       
       xmin = 0; % minimum value x can take during the episode
       xmax = 10; % maximum value x can take during the episode
       
       L = 0; % L is set the constructor
       init_xvals = [];
       Figure = [];
    end
    
    properties
        % Initialize system state [x,dx,y,dy]'
        X = zeros(2,1);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        curStep = 0;
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = TreeClassCustomizable_xy(type, n, w)
           
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([2 1]);
            ObservationInfo.Name = 'Tree system states';
            ObservationInfo.Description = 'x, y';
            
            % Initialize Action settings   
            L = 5;
            ActionInfo = rlNumericSpec([1], 'LowerLimit', -5, 'UpperLimit', 5);
            ActionInfo.Name = 'Tree system Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.L = L;
            this.init_xvals = [1:this.L*this.dt:9];
            
            % This part initiates the trees depending in the inputs
            if ~exist('type','var') || isempty(type)
                type = 'default';
            end
            if ~exist('n','var') || isempty(n) 
                if ~exist('w','var') || isempty(w)
                    w = this.w_obs;
                end
                n = (this.xmax-this.xmin)/w;
            end
            if ~exist('w','var') || isempty(w)
                w = this.w_obs;
            end
            
            % decide what type of env
            if strcmp(type,'default') % single hardcoded obstacle
                d = this.deadzones;    
            elseif strcmp(type,'linear')
                linearObstacles(this,n,w);
            elseif strcmp(type,'endor')
                endor(this,n,w);
            end
                
           
            d = this.deadzones;
            
            
            %plot obstacles when the env is created
            %figure()
            %plot(d(:,1:2)', [d(:,3) d(:,3)]','r','LineWidth',1)
            %axis([0 10 -10 2])
            % Initialize property values and pre-compute necessary values
            % this.ActionInfo.Elements = this.MaxForce*[-1 1];

        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];
            Action = max(-this.L, Action);
            Action = min(this.L, Action);
            
            x = this.X(1);
            x = x + Action*this.dt;
            x = max(this.xmin, x);
            x = min(this.xmax, x);
            
            y = this.X(2);
            y = y + this.g*this.dt;
            
            this.X(1) = x;
%             this.X(2) = Action;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            this.X(2) = y;
%             this.X(4) = this.g; 
            Observation = this.X;
            
            Reward = 0;
            %Reward = Reward - 0.01*(Action.^2) + 1;
            %term =(x>=this.deadzone(1)) & (x<=this.deadzone(2)) & (-this.tol < y) & (y < this.tol);
            term = any(x>=this.deadzones(:,1)) && any(x<=this.deadzones(:,2)) && any(-this.tol<(y-this.deadzones(:,3))&(y-this.deadzones(:,3))<this.tol); % determines if there is a collision with any of the trees 

            Reward = Reward -25*term;
           
            IsDone = this.curStep >= this.N || term;
            this.curStep = this.curStep + 1;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            this.X(2) = this.yinit;
            this.X(1) = randsample(this.init_xvals,1);
%             this.X(2) = 0;
%             this.X(4) = this.g;
            InitialObservation = this.X;
           
            this.curStep = 0; 
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods   
        %function to create a line of obstacles
        function linearObstacles(this,n,w)
            this.n_obs = n;
            this.w_obs = w;
            if n >= (this.xmax-this.xmin)/w
                n=n-1;
            end
            lim = [this.xmin,this.xmax];
            minDist = .1+w;
            mid_pts = nan(n,1);
            c = 0;
            while any(isnan(mid_pts(:))) && c<(n*3000)
                % Fill NaN values with new random coordinates
                mid_pts(isnan(mid_pts)) = rand(1,sum(isnan(mid_pts(:)))) * (lim(2)-lim(1)) + lim(1);
                % Identify rows that are too close to another point
                [~,isTooClose] = find(triu(squareform(pdist(mid_pts)) < minDist,1));
                % Replace too-close coordinates with NaN
                mid_pts(isTooClose,:) = NaN; 
                c = c+1;
            end

            d = [mid_pts-w/2 mid_pts+w/2 zeros(length(mid_pts),1)];
            this.deadzones = d;
        end
        
        % fuction to create the forest
        function endor(this,n,w)
            this.n_obs = n;
            this.w_obs = w;
            ylims = [-1 5];
            lim = [this.xmin,this.xmax];
            minDist = .1+w;
            d = [];
            for rows = 1:ylims(2)
                mid_pts = nan(n,1);
                c=0;
                while any(isnan(mid_pts(:))) && c<(n*3000)
                    % Fill NaN values with new random coordinates
                    mid_pts(isnan(mid_pts)) = rand(1,sum(isnan(mid_pts(:)))) * (lim(2)-lim(1)) + lim(1);
                    % Identify rows that are too close to another point
                    [~,isTooClose] = find(triu(squareform(pdist(mid_pts)) < minDist,1));
                    % Replace too-close coordinates with NaN
                    mid_pts(isTooClose,:) = NaN; 
                    c = c+1;
                end
%                 d = [d; mid_pts-w/2 mid_pts+w/2 ones(n,1)*rows];     % use this line for uniform rows of obstacles
                d = [d; mid_pts-w/2 mid_pts+w/2 -rows+(rand(n,1)*0.5-.25)];  % use this line for a random offset in y coordinate of obstacles
            end
            d(any(isnan(d), 2), :) = []; % remove rows that has NaN
            this.deadzones = d;
        end
        % Helper methods to create the environment
        % Discrete force 1 or 2
%         function force = getForce(this,action)
%             if ~ismember(action,this.ActionInfo.Elements)
%                 error('Action must be %g for going left and %g for going right.',-this.MaxForce,this.MaxForce);
%             end
%             force = action;           
%         end
%         % update the action info based on max force
%         function updateActionInfo(this)
%             this.ActionInfo.Elements = this.MaxForce*[-1 1];
%         end
%         
%         % Reward function
%         function Reward = getReward(this)
%             if ~this.IsDone
%                 Reward = this.RewardForNotFalling;
%             else
%                 Reward = this.PenaltyForFalling;
%             end          
%         end
%         
%         % (optional) Visualization method
        function plot(this)
            this.Figure = figure('Visible','on','HandleVisibility','off');
            ha = gca(this.Figure);
            ha.XLimMode = 'manual';
            ha.YLimMode = 'manual';
            ha.XLim = [0 10];
            ha.YLim = [-10 2];
            hold(ha,'on');
            % Update the visualization
            envUpdatedCallback(this)
        end
        
        % (optional) Properties validation through set methods
%         function set.State(this,state)
%             validateattributes(state,{'numeric'},{'finite','real','vector','numel',4},'','State');
%             this.State = double(state(:));
%             notifyEnvUpdated(this);
%         end
%         function set.HalfPoleLength(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','HalfPoleLength');
%             this.HalfPoleLength = val;
%             notifyEnvUpdated(this);
%         end
%         function set.Gravity(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Gravity');
%             this.Gravity = val;
%         end
%         function set.CartMass(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','CartMass');
%             this.CartMass = val;
%         end
%         function set.PoleMass(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','PoleMass');
%             this.PoleMass = val;
%         end
%         function set.MaxForce(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','MaxForce');
%             this.MaxForce = val;
%             updateActionInfo(this);
%         end
%         function set.Ts(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
%             this.Ts = val;
%         end
%         function set.AngleThreshold(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','AngleThreshold');
%             this.AngleThreshold = val;
%         end
%         function set.DisplacementThreshold(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','DisplacementThreshold');
%             this.DisplacementThreshold = val;
%         end
%         function set.RewardForNotFalling(this,val)
%             validateattributes(val,{'numeric'},{'real','finite','scalar'},'','RewardForNotFalling');
%             this.RewardForNotFalling = val;
%         end
%         function set.PenaltyForFalling(this,val)
%             validateattributes(val,{'numeric'},{'real','finite','scalar'},'','PenaltyForFalling');
%             this.PenaltyForFalling = val;
%         end
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
            if ~isempty(this.Figure) && isvalid(this.Figure)
                % Set visualization figure as the current figure
                
                
                ha = gca(this.Figure);
                figure(this.Figure)
%                 delete(gca)
                d = this.deadzones;
                % Extract the position
                x = this.X(1);
                y = this.X(2);
                agntplot = findobj(ha,'Tag','agentplot');
                if isempty(agntplot) || ~isvalid(agntplot)
                    agnt = polyshape([-0.1 -0.1 0.1 0.1],[-0.1 0.1 0.1 -0.1]);
%                     agnt = rectangle(ha,'Position',[x-0.075 y-0.075 0.15 0.15], 'Curvature', 1, 'FaceColor', [0, 1, 1, 0.3]); 
                    agnt = translate(agnt, [x,y]);
                    agntplot = plot(ha,agnt);
                    
                    agntplot.Tag = 'agentplot';
                else
                    agnt = agntplot.Shape;
                end
                [agntposx,agntposy] = centroid(agnt);
                dx = x - agntposx;
                dy = y - agntposy;
                agnt = translate(agnt, [dx,dy]);
                agntplot.Shape = agnt;
%                 agnt = rectangle(ha,'Position',[x-0.075 y-0.075 0.15 0.15], 'Curvature', 1, 'FaceColor', [0, 1, 1, 0.3]);
%                 agnt.Position = [x-0.075 y-0.075 0.15 0.15];
                obsplot = plot(ha,d(:,1:2)', [d(:,3) d(:,3)]','r','LineWidth',2);
%                 hold on
%                 agntplot.Shape = agnt;
                
                               
                
                % Refresh rendering in the figure window
                drawnow();
            end
        end
    end
end
