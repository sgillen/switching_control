clear all;
env = TreeClassCustomizable_xy;
initOpts = rlAgentInitializationOptions('NumHiddenUnit',4);
agentOpts = rlSACAgentOptions('UseDeterministicExploitation', true,'NumStepsToLookAhead',env.N);
agent = rlSACAgent(env.getObservationInfo,env.getActionInfo,initOpts,agentOpts);

%agent = rlSACAgent(env.getObservationInfo,env.getActionInfo,initOpts);

%%%%%%%%%
% critic = getCritic(agent);
% % critic.Options.LearnRate = 5*1e-3;
% critic.Options.UseDevice = 'cpu';
% agent  = setCritic(agent,critic);
% % %%%%%%%%%%
% actor = getActor(agent);
% actor.Options.UseDevice = 'cpu';
% % actor.Options.LearnRate = 1e-3;
% agent  = setActor(agent,actor);

opt = rlTrainingOptions('MaxEpisodes',3000,'MaxStepsPerEpisode',env.N,'ScoreAveragingWindowLength',100);
trainstats = train(agent, env, opt);

agent.AgentOptions.UseDeterministicExploitation = true;
visualizeAgent(agent, env);