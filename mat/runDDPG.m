clear all;
env = TreeClassCustomizable_test;
initOpts = rlAgentInitializationOptions('NumHiddenUnit',64);
agent = rlDDPGAgent(env.getObservationInfo,env.getActionInfo,initOpts);
%%%%%%%%%
critic = getCritic(agent);
% critic.Options.LearnRate = 5*1e-3;
critic.Options.UseDevice = 'cpu';
agent  = setCritic(agent,critic);
% %%%%%%%%%%
actor = getActor(agent);
actor.Options.UseDevice = 'cpu';
% actor.Options.LearnRate = 1e-3;
agent  = setActor(agent,actor);

opt = rlTrainingOptions('MaxEpisodes',700,'MaxStepsPerEpisode',60,'ScoreAveragingWindowLength',100);
trainstats = train(agent, env, opt);
