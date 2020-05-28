from env import *
from collections import deque

from env import *
from worker import RolloutWorker
from agent import Agents
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer

args = get_common_args()
args = qmix_args(args)

env = RandomEnv(args)
agents = Agents(args)
worker = RolloutWorker(env, agents, args)
buffer = ReplayBuffer(args)

plt.figure()
plt.axis([0, args.n_epoch, 0, 100])
win_rates = []
episode_rewards = []
train_steps = 0

save_path = args.result_dir + '/' + args.alg

def evaluate():
    win_number = 0
    episode_rewards = 0
    for epoch in range(args.n_evaluate_episode):
        _, episode_reward = worker.generate_episode(evaluate=True)
        episode_rewards += episode_reward
        if episode_reward > args.threshold:
            win_number += 1
    return win_number / args.n_evaluate_episode, episode_rewards / args.n_evaluate_episode

for epoch in range(args.n_epoch):
    print('Run {}, train epoch {}'.format(1, epoch))

    if epoch % args.evaluate_cycle == 0:
        win_rate, episode_reward = evaluate()
        win_rates.append(win_rate)
        episode_rewards.append(episode_reward)
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(win_rates)), win_rates)
        plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(save_path + '/plt_{}.png'.format(1), format='png')
        np.save(save_path + '/win_rates_{}'.format(1), win_rates)
        np.save(save_path + '/episode_rewards_{}'.format(1), episode_rewards)

    episodes = []
    for episode_idx in range(args.n_episodes):
        episode, _ = worker.generate_episode(episode_idx)
        episodes.append(episode)
    episode_batch = episodes[0]
    episodes.pop(0)
    for episode in episodes:
        for key in episode_batch.keys():
            episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

    if args.alg.find('coma') > -1 or args.alg.find('central_v') > -1 or args.alg.find('reinforce') > -1:
        agents.train(episode_batch, train_steps, worker.epsilon)
        train_steps += 1
    else:
        buffer.store_episode(episode_batch)
        for train_step in range(args.train_steps):
            mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
            agents.train(mini_batch, train_steps)
            train_steps += 1

plt.cla()
plt.subplot(2, 1, 1)
plt.plot(range(len(win_rates)), win_rates)
plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
plt.ylabel('win_rate')

plt.subplot(2, 1, 2)
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
plt.ylabel('episode_rewards')

plt.savefig(save_path + '/plt_{}.png'.format(1), format='png')
np.save(save_path + '/win_rates_{}'.format(1), win_rates)
np.save(save_path + '/episode_rewards_{}'.format(1), episode_rewards)


