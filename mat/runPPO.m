clear all;
env = TreeClass;
agent = rlPPOAgent(env.getObservationInfo,env.getActionInfo);
opt = rlTrainingOptions('MaxEpisodes',1000,'MaxStepsPerEpisode',50);
trainstats = train(agent, env, opt);