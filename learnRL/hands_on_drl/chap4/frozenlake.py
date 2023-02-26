import gymnasium as gym
import numpy as np
from collections import namedtuple
from torch import nn
from torch import optim
import torch

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=0)
    while True:
        obs_v = torch.FloatTensor(obs)
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.detach().numpy()
        action = np.random.choice(act_probs.size, p=act_probs)
        next_obs, reward, is_done, _, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    disc_rewards = [e.reward * (GAMMA ** len(e.steps)) for e in batch]
    reward_bound = np.percentile(disc_rewards, percentile)
    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward >= reward_bound:
            train_obs.extend((step.observation for step in example.steps))
            train_act.extend((step.action for step in example.steps))
            elite_batch.append(example)
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return elite_batch, train_obs_v, train_act_v, reward_bound


if __name__ == "__main__":
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1", is_slippery=False))
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env,net, BATCH_SIZE)):
        reward_m = float(np.mean([e.reward for e in batch]))
        full_batch, obs_v, acts_v, reward_b = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        full_batch = full_batch[-500:]

        action_scores_v = net(obs_v)
        loss_v = loss_fn(action_scores_v, acts_v)

        optimizer.zero_grad()
        loss_v.backward()
        optimizer.step()

        print(f"{iter_no}: loss={loss_v.item(): .3f}, reward_mean={reward_m: .3f}, reward_bound={reward_b: .3f}")

        if reward_m > 0.8:
            print("Solved!")
            break