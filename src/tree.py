# %%

import gym
import numpy as np
import seagul.envs

env = gym.make('tree_simple-v0')
env.reset()
env.step(env.action_space.sample())
env.step(env.action_space.sample())

# %%

from seagul.rl.sac import SACAgent, SACModel
from seagul.nn import MLP

# %%
# ===========================
#
# policy_net = MLP(env.observation_space.shape[0], env.action_space.shape[0] * 2, 2, 64, input_bias=True)
# value_net = MLP(env.observation_space.shape[0], 1, 2, 64, input_bias=True)
# q1_net = MLP(env.observation_space.shape[0] + env.action_space.shape[0], 1, 2, 64, input_bias=True)
# q2_net = MLP(env.observation_space.shape[0] + env.action_space.shape[0], 1, 2, 64, input_bias=True)
#
# model = SACModel(policy_net, value_net, q1_net, q2_net, 5)
# agent = SACAgent('tree_simple-v0', model, env_max_steps=5)

# ===========================
#
# from seagul.rl.ppo import PPOAgent, PPOModel
# from seagul.nn import MLP
#
# policy_net = MLP(env.observation_space.shape[0], env.action_space.shape[0] , 2, 64, input_bias=True)
# value_net = MLP(env.observation_space.shape[0], 1, 1, 64, input_bias=True)
#
# model = PPOModel(policy_net, value_net)
# agent = PPOAgent('tree_simple-v0', model, env_no_term_steps=5)

# ===========================

from seagul.rl.ars import ARSAgent

agent = ARSAgent('tree_multi-v0', 0, n_delta=256, n_top=256, step_size=0.05, exp_noise=0.05)

# ===========================

# %%

agent.learn(10000)

# %%

import matplotlib.pyplot as plt
from matplotlib import cm
plt.plot(agent.lr_hist)
plt.show()

# %%
x = np.linspace(0,10, 200)
y = np.linspace(-1,2,100)

U = np.zeros([x.shape[0], y.shape[0]])

for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        #U[i,j] = agent.model.policy(np.array([x[i], y[j]], dtype=np.float32))[0]
        U[i,j],_,_,_ = agent.model.step(np.array([x[i], y[j]], dtype=np.float32))

U = np.clip(U, -5, 5);

# %%
X,Y = np.meshgrid(x,y, indexing='ij')
XW, ZW = np.meshgrid(range(3,7), range(-5,5), indexing='ij')
YW = XW*0

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.plot_surface(X,Y,U, facecolors=cm.Spectral(U/np.amax(U)), alpha=.5)
ax.plot_surface(XW,YW,ZW, color='blue', alpha = .5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')
plt.show()

