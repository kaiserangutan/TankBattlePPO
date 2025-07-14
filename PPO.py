
import os
import random
import gym
from collections import deque, namedtuple
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from TankBattleEnv_05 import TankBattleEnv
from functools import partial

LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.21
ENT_COEF = 0.01
VF_COEF = 0.5
ROLL_OUT_STEPS = 3000
BATCH_SIZE = 250
PPO_EPOCHS = 10
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cuda")


# Simple container for rollout data
Transition = namedtuple(
    "Transition",
    ["obs", "action", "logprob", "reward", "done", "value"],
)


class PolicyNetwork(nn.Module):


    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 128),
            nn.Tanh(),
            nn.Linear(128, 96),
            nn.Tanh(),
            nn.Linear(96, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape (...,)


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int):
        self.actor = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
        self.critic = ValueNetwork(obs_dim).to(DEVICE)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.rollout: List[Transition] = []
        self.kl_history: List[float] = [] 
        self._training_step = 0
    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = self.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(obs_t)
        return (
            int(action.item()),
            float(logprob.cpu().item()),
            float(value.cpu().item()),
        )

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.rollout.append(
            Transition(obs, action, logprob, reward, done, value)
        )

    def _compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns advantage and returns tensors."""
        advantages = []
        gae = 0.0
        for step in reversed(self.rollout):
            if step.done:
                delta = step.reward - step.value
                gae = delta
            else:
                delta = step.reward + GAMMA * next_value - step.value
                gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages.insert(0, gae)
            next_value = step.value
        returns = [adv + t.value for adv, t in zip(advantages, self.rollout)]
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return adv_t, ret_t

    def update(self, next_value: float):
        if len(self.rollout) == 0:
            return

        obs = torch.tensor(
            np.array([t.obs for t in self.rollout]), dtype=torch.float32, device=DEVICE
        )
        actions = torch.tensor(
            np.array([t.action for t in self.rollout]), dtype=torch.long, device=DEVICE
        )
        old_logprobs = torch.tensor(
            np.array([t.logprob for t in self.rollout]), dtype=torch.float32, device=DEVICE
        )

        with torch.no_grad():
            new_logits = self.actor(obs)
            new_dist   = torch.distributions.Categorical(logits=new_logits)
            new_logprobs = new_dist.log_prob(actions)

        kl = (old_logprobs - new_logprobs).mean().item()
        self.kl_history.append(kl) 

        advantages, returns = self._compute_gae(next_value)

        dataset_size = len(self.rollout)
        idxs = np.arange(dataset_size)

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = idxs[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_oldlog = old_logprobs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]

                logits = self.actor(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(logprobs - batch_oldlog)
                surr1 = ratios * batch_adv
                surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean() - ENT_COEF * entropy

                values = self.critic(batch_obs)
                critic_loss = F.mse_loss(values, batch_ret) * VF_COEF

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optim.step()

        self.rollout.clear()
        self._training_step += 1


    def save_checkpoint(self, path: str):
        """Save actor/critic parameters and optimizer states."""
        checkpoint = {
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "step": self._training_step,
        }
        torch.save(checkpoint, path)
        print(f"[PPOAgent] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, map_location: str = "cpu"):
        """Load all parameters/optimizer states from disk."""
        checkpoint = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim"])
        self._training_step = checkpoint.get("step", 0)
        print(f"[PPOAgent] Checkpoint loaded from {path}")


def train(
    num_episodes: int = 500,
    checkpoint_dir: str = "./checkpoints",
    save_every: int = 40,
    checkpoint_to_load: str = "",
    rand: bool = False
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = TankBattleEnv(render_mode=None)
    obs, enemy_obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim)
    enemy = PPOAgent(obs_dim, act_dim)
    
    if len(checkpoint_to_load):
        agent.load_checkpoint(checkpoint_to_load)
    episode_rewards = []
    episode_durations = []
    win = 0
    lose = 0
    __a = 0
    for ep in range(num_episodes):
        if ep % 50 == 0 and not rand:
            try:
                enemy.load_checkpoint("./checkpoints/" + str(np.random.randint(1, 11)) + ".pt")
            except:
                if len(checkpoint_to_load):
                    enemy.load_checkpoint(checkpoint_to_load)
                else:
                    pass
        ep_reward = 0
        timestep = 0
        done = False
        # env.render_mode = "human"
        while not done:

            action, logprob, value = agent.select_action(obs)
            if rand:
                enemy_action = env.action_space.sample()
            else:
                enemy_action, _, _= enemy.select_action(enemy_obs)
            
            next_obs, next_enemy_obs, reward, done, _ = env.step(action, enemy_action)
            timestep += 1
            ep_reward += reward

            agent.store_transition(obs, action, logprob, reward, done, value)

            if len(agent.rollout) >= ROLL_OUT_STEPS:
                with torch.no_grad():
                    _, _, next_value = agent.select_action(next_obs)
                agent.update(next_value)

            obs = next_obs
            enemy_obs = next_enemy_obs

        episode_rewards.append(ep_reward)
        episode_durations.append(timestep)
        kl_history = agent.kl_history
        win += 1 if ep_reward > 950 else 0
        lose += 1 if ep_reward < -950 else 0
        
        obs, enemy_obs, _ = env.reset()
        if len(agent.rollout):
            agent.update(0.0)
        
        agent.rollout.clear()

        print("Episode", ep, ep_reward, timestep)
        ep_reward = 0

        

        if (ep // save_every) != ((ep - 1) // save_every):
            ckpt_path = os.path.join(checkpoint_dir, f"{__a % 10 + 1}.pt") # keep most recent 10 checkpoints
            agent.save_checkpoint(ckpt_path)
            __a += 1
    print("Episodes:", num_episodes, "; Wins:", win, "Draws:", num_episodes - win - lose, "Loss:", lose)

    # final save
    agent.save_checkpoint(os.path.join(checkpoint_dir, "final.pt"))
    env.close()
    avg_rewards = []
    avg_durations = []
    for i in range(len(episode_rewards) - 20):
        avg_rewards.append(np.average(episode_rewards[i:i + 20]))
    for i in range(len(episode_durations) - 20):
        avg_durations.append(np.average(episode_durations[i:i + 20]))
    return avg_rewards, avg_durations, kl_history

def visualize(enemy_checkpoint_path, checkpoint_path: str = "./checkpoints/final.pt", seed: int = None, rand = False):

    env = TankBattleEnv(render_mode="human")
    obs, enemy_obs, _ = env.reset(seed=seed)

    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load_checkpoint(checkpoint_path, map_location=DEVICE)
    if not rand:
        print("Loading enemy checkpoint")
        enemy = PPOAgent(env.observation_space.shape[0], env.action_space.n)
        enemy.load_checkpoint(enemy_checkpoint_path, map_location=DEVICE)


    total_reward, done = 0.0, False
    while not done:
        action, _, _ = agent.select_action(obs)
        if not rand:
            enemy_action, _, _ = enemy.select_action(enemy_obs)
        else:
            enemy_action = env.action_space.sample()
        obs, enemy_obs, reward, done, _ = env.step(action, enemy_action)
        total_reward += reward
        # print(enemy_action)

    print(f"Episode finished — return: {total_reward:.1f}")
    env.close()
    return total_reward

def compare(enemy_checkpoint_path, checkpoint_path: str = "./checkpoints/final.pt", seed: int = None, rand = False, rounds = 200):

    env = TankBattleEnv(render_mode=None)
    obs, enemy_obs, _ = env.reset(seed=seed)
    win = 0
    lose = 0

    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load_checkpoint(checkpoint_path, map_location=DEVICE)
    if not rand:
        print("Loading enemy checkpoint")
        enemy = PPOAgent(env.observation_space.shape[0], env.action_space.n)
        enemy.load_checkpoint(enemy_checkpoint_path, map_location=DEVICE)
    for i in range(rounds):
        env.reset(seed=seed)
        total_reward = 0.0
        done = False
        while not done:
            action, _, _ = agent.select_action(obs)
            if not rand:
                enemy_action, _, _ = enemy.select_action(enemy_obs)
            else:
                enemy_action = env.action_space.sample()
            obs, enemy_obs, reward, done, _ = env.step(action, enemy_action)
            total_reward += reward
        if total_reward > 1000:
            win += 1
        elif total_reward < -1000:
            lose += 1
        # print(total_reward)

    print(f"Competition over. Agent wins:",win, "; Enemy wins:",lose, "Draws:",rounds - win - lose)
    env.close()


def plot_training_curves(rewards, lengths, kl_list):
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].plot(rewards)
    axes[0].set_title('Average Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')

    axes[1].plot(lengths)
    axes[1].set_title('Average Episode Length')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')

    axes[2].plot(kl_list)
    axes[2].set_title('Mean KL Div')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps')


    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # avg_rewards, avg_durations, kl_list= train(num_episodes=1000, checkpoint_to_load="./checkpoints/self_1000.pt", rand = False); plot_training_curves(avg_rewards, avg_durations, kl_list); 
    visualize(enemy_checkpoint_path="./checkpoints/random_500.pt", checkpoint_path= "./checkpoints/self_2000.pt", rand=True)
    # compare(enemy_checkpoint_path="./checkpoints/random_500.pt", checkpoint_path= "./checkpoints/self_2000.pt", rand=False)
    