import numpy as np
import torch
from qmix import QMIX
from torch.distributions import Categorical

class Agents:
    def __init__(self, args):
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        self.policy = QMIX(args)
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]

        agent_id = np.zeros(self.num_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))

        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        # avail_actions =torch. tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # q_value[avail_actions == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            # action = np.random.choice(avail_actions_ind)
            action = np.random.choice(self.args.num_actions)
        else:
            action = torch.argmax(q_value)

        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]

        max_episode_len = 0

        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.max_episode_steps):
                if transition_idx + 1 >= max_episode_len:
                    max_episode_len = transition_idx + 1
                break

        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        max_episdoe_len = self._get_max_episode_len(batch)

        for key in batch.keys():
            batch[key] = batch[key][:,:max_episdoe_len]

        self.policy.learn(batch, max_episdoe_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)


