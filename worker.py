import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        # self.episode_limit = args.episode_limit
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        print('init rollout worker')

    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False

        step = 0
        episode_reward = 0

        last_action = np.zeros((self.args.num_agents, self.args.num_actions))
        self.agents.policy.init_hidden(1)


        #epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()

            """
            State and Observations
            At each timestep, agents receive local observations drawn within their field of view. This encompasses information about the map within a circular area around each unit and with a radius equal to the sight range. The sight range makes the environment partially observable from the standpoint of each agent. Agents can only observe other agents if they are both alive and located within the sight range. Hence, there is no way for agents to determine whether their teammates are far away or dead.
            
            The feature vector observed by each agent contains the following attributes for both allied and enemy units within the sight range: distance, relative x, relative y, health, shield, and unit_type 1. Shields serve as an additional source of protection that needs to be removed before any damage can be done to the health of units. All Protos units have shields, which can regenerate if no new damage is dealt (units of the other two races do not have this attribute). In addition, agents have access to the last actions of allied units that are in the field of view. Lastly, agents can observe the terrain features surrounding them; particularly, the values of eight points at a fixed radius indicating height and walkability.
            
            The global state, which is only available to agents during centralised training, contains information about all units on the map. Specifically, the state vector includes the coordinates of all agents relative to the centre of the map, together with unit features present in the observations. Additionally, the state stores the energy of Medivacs and cooldown of the rest of allied units, which represents the minimum delay between attacks. Finally, the last actions of all agents are attached to the central state.
            
            All features, both in the state as well as in the observations of individual agents, are normalised by their maximum values. The sight range is set to 9 for all agents.
            """
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.num_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                avail_action = None
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                action_onehot = np.zeros(self.args.num_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                # avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, _ = self.env.step(actions)
            if step == self.args.max_episode_steps - 1:
                terminated = 1

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.num_agents, 1]))
            u_onehot.append(actions_onehot)
            # avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        # 最后一个obs需要单独计算一下avail_action，到时候需要计算target_q
        # avail_actions = []
        # for agent_id in range(self.num_agents):
            # avail_action = self.env.get_avail_agent_actions(agent_id)
            # avail_actions.append(avail_action)
        # avail_u.append(avail_actions)
        # avail_u_next = avail_u[1:]
        # avail_u = avail_u[:-1]

        for i in range(step, self.args.max_episode_steps):  # 没有的字段用0填充，并且padded为1
            o.append(np.zeros((self.num_agents, self.obs_space)))
            u.append(np.zeros([self.num_agents, 1]))
            s.append(np.zeros(self.state_space))
            r.append([0.])
            o_next.append(np.zeros((self.num_agents, self.obs_space)))
            s_next.append(np.zeros(self.state_space))
            u_onehot.append(np.zeros((self.num_agents, self.num_actions)))
            # avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            # avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        '''
        (o[n], u[n], r[n], o_next[n], avail_u[n], u_onehot[n])组成第n条经验，各项维度都为(episode数，transition数，n_agents, 自己的具体维度)
         因为avail_u表示当前经验的obs可执行的动作，但是计算target_q的时候，需要obs_net及其可执行动作，
        '''
        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       # avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )

        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        return episode, episode_reward

