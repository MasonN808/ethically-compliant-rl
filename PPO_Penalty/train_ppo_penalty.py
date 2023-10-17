#!/usr/bin/env python3
"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

View more on my tutorial website: https://morvanzhou.github.io/tutorials
"""

import ast
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import load_environment
import wandb
import tensorflow_probability as tfp

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

seed = 10

# Set the numpy seed
np.random.seed(seed)

# Set the pytorch seed
# Set the seed for CPU
torch.manual_seed(seed)

# Initialize wandb
# wandb.init(name="ppo-highway-parking", project="PPO", sync_tensorboard=True)

with open('configs/ParkingEnv/env-default.txt') as f:
    data = f.read()

# Reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
# Overriding certain keys in the environment config
ENV_CONFIG.update({
    "start_angle": -np.math.pi/2, # This is radians
})

class PPO(object):

    def __init__(self):
        self.tfs = tf.constant(np.zeros((1, S_DIM), dtype=np.float32), dtype=tf.float32)

        # critic
        self.critic = self.build_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=C_LR)

        # actor
        self.actor = self.build_actor()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=A_LR)

    def build_critic(self):
        critic = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return critic

    def build_actor(self):
        inputs = tf.keras.layers.Input(shape=(S_DIM,))
        x = tf.keras.layers.Dense(100, activation='relu')(inputs)
        mu = 2 * tf.keras.layers.Dense(A_DIM, activation='tanh')(x)
        sigma = tf.keras.layers.Dense(A_DIM, activation='softplus')(x)
        actor = tf.keras.Model(inputs, [mu, sigma])
        return actor

    def ppo_loss(self, y_true, y_pred):
        old_mu, old_sigma = y_true[:, :A_DIM], y_true[:, A_DIM:]
        mu, sigma = y_pred[:, :A_DIM], y_pred[:, A_DIM:]
        td_error = tf.reduce_mean(self.advantage() * self.old_log_prob() / (old_sigma + 1e-5))
        return -td_error

    def advantage(self):
        return self.tfdc_r - self.critic(self.tfs)

    def old_log_prob(self):
        old_mu, old_sigma = self.oldpi()
        old_dist = tfp.distributions.Normal(loc=old_mu, scale=old_sigma)
        return old_dist.log_prob(self.tfa)

    def oldpi(self):
        mu, sigma = self.actor(self.tfs)
        old_mu, old_sigma = tf.stop_gradient(mu), tf.stop_gradient(sigma)
        return old_mu, old_sigma

    def choose_action(self, s):
        s = s[np.newaxis, :]
        mu, sigma = self.actor(s)
        a = np.clip(np.random.normal(mu, sigma), -2, 2)
        return a

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.critic(s)[0, 0]

    def update(self, s, a, r):
        with tf.GradientTape() as tape:
            v_s_ = self.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]

            # actor loss
            mu, sigma = self.actor(bs)
            old_mu, old_sigma = self.oldpi()
            ratio = tf.exp(self.actor.log_prob(ba) - self.old_log_prob())
            surr = ratio * self.advantage()
            if METHOD['name'] == 'kl_pen':
                kl = tfp.distributions.kl_divergence(
                    tfp.distributions.Normal(loc=old_mu, scale=old_sigma),
                    tfp.distributions.Normal(loc=mu, scale=sigma)
                )
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - METHOD['lam'] * kl))
            else:
                aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.advantage()))

            # critic loss
            closs = tf.reduce_mean(tf.square(self.advantage()))

            total_loss = aloss + closs
            gradients = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))

if __name__ == "__main__":
    env = FlattenObservation(load_environment(ENV_CONFIG))
    env = DummyVecEnv([lambda: env])
    ppo = PPO()
    all_ep_r = []

    for ep in range(EP_MAX):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):    # in one episode
            env.render()
            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r+8)/8)    # normalize reward, find to be useful
            s = s_
            ep_r += r

            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1:
                ppo.update(buffer_s, buffer_a, buffer_r)
                buffer_s, buffer_a, buffer_r = [], [], []
        if ep == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()
