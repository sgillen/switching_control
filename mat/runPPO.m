clear all;
env = TreeClassCustomizable_xy;
initOpts = rlAgentInitializationOptions('NumHiddenUnit',16);
agentOpts = rlPPOAgentOptions('UseDeterministicExploitation', true);
agent = rlPPOAgent(env.getObservationInfo,env.getActionInfo,initOpts,agentOpts);

%%%%%%%%%
critic = getCritic(agent);
critic.Options.LearnRate = 5*1e-3;
critic.Options.UseDevice = 'cpu';
agent  = setCritic(agent,critic);
%%%%%%%%%%
actor = getActor(agent);
actor.Options.UseDevice = 'cpu';
actor.Options.LearnRate = 1e-3;
agent  = setActor(agent,actor);

visualizeAgent(agent,env);
agent.AgentOptions.UseDeterministicExploitation = false;
visualizeAgent(agent,env);


opt = rlTrainingOptions('MaxEpisodes',5000,'MaxStepsPerEpisode',env.N,'ScoreAveragingWindowLength',100);
trainstats = train(agent, env, opt);

visualizeAgent(agent,env);
agent.AgentOptions.UseDeterministicExploitation = false;
visualizeAgent(agent,env);

