classdef TreeClass < rl.env.MATLABEnvironment
    %TREECLASS: Template for defining custom environment in MATLAB.    
    
    %% Properties 
    properties
       g = -5 % The constant acceleration in y.
        yinit = 2; % yvalue to inizialize y at
       dt = .01; 
       tol = 0.01; % tolerance for accepting that we are in a deadzone
       N = 50; % how many steps i n an episode
       deadzone = [3,7]; % where does the tree live?
       xmin = 0; % minimum value x can take during the episode
       xmax = 10; % maximum value x can take during the episode
       
       L = 0; % L is set the constructor
       init_xvals = [];
    end
    
    properties
        % Initialize system state [x,dx,theta,dtheta]'
        X = zeros(2,1);
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
        function this = TreeClass()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([2 1]);
            ObservationInfo.Name = 'Tree system states';
            ObservationInfo.Description = 'x, y';
            
            % Initialize Action settings   
            L = 10;
            ActionInfo = rlNumericSpec([1], 'LowerLimit', -5, 'UpperLimit', 5);
            ActionInfo.Name = 'Tree system Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.L = L;
            this.init_xvals = [0:this.L*this.dt:10];

            
            % Initialize property values and pre-compute necessary values
             %this.ActionInfo.Elements = this.MaxForce*[-1 1];

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
            x = min(this.xmax,x);
            
            y = this.X(2);
            y = y + this.g*this.dt;
            
            this.X(1) = x;
            this.X(2) = y;
            Observation = this.X;
            
            Reward = 0;
            Reward = Reward - (Action.^2);
            term =(x>=this.deadzone(1)) & (x<=this.deadzone(2)) & (-this.tol < y) & (y < this.tol);
            Reward = Reward -10000*term;
           
            IsDone = this.curStep >= this.N || term;
            this.curStep = this.curStep + 1;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            this.X(1) = randsample(this.init_xvals,1);
            this.X(2) = this.yinit;
            InitialObservation = this.X;
           
            this.curStep = 0; 
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
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
%         function plot(this)
%             % Initiate the visualization
%             
%             % Update the visualization
%             envUpdatedCallback(this)
%         end
        
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
        end
    end
end
