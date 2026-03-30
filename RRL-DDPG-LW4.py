# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:43:25 2026

@author: swaro
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# ============================
# Hyperparameters
# ============================
ENV_NAME = "Pendulum-v1"
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
BUFFER_SIZE = 100000
BATCH_SIZE = 64
EPISODES = 150
MAX_STEPS = 200

# ============================
# Replay Buffer
# ============================
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

    def size(self):
        return len(self.buffer)

# ============================
# Actor Network
# ============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action

# ============================
# Critic Network
# ============================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# ============================
# DDPG Agent
# ============================
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.max_action = max_action
        self.buffer = ReplayBuffer()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).detach().numpy()[0]

    def train(self):
        if self.buffer.size() < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample()

        # Critic loss
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + (1 - dones) * GAMMA * target_Q.detach()

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# ============================
# Training Loop
# ============================
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim, max_action)

rewards_history = []

for ep in range(EPISODES):
    state, _ = env.reset()
    ep_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)

        # Add exploration noise
        action = (action + np.random.normal(0, 0.2, size=action_dim)).clip(-max_action, max_action)

        next_state, reward, done, truncated, _ = env.step(action)

        agent.buffer.add((state, action, reward, next_state, float(done or truncated)))

        agent.train()

        state = next_state
        ep_reward += reward

        if done or truncated:
            break

    rewards_history.append(ep_reward)
    print(f"Episode {ep+1}, Reward: {ep_reward:.2f}")

# ============================
# Plot Results
# ============================
plt.plot(rewards_history)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("DDPG Training Performance")
plt.show()