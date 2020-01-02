import gym
import torch
from agent import VPG


def main():
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)
    episodes = 1000
    running_reward = 10
    gamma = 0.99
    learning_rate = 0.01

    agent = VPG(env, gamma=gamma, learning_rate=learning_rate)

    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False

        for time in range(10000):
            if episode % 50 == 0:
                env.render()
            action = agent.select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action)

            # Save reward
            agent.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        agent.update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                        time))
            break


if __name__ == '__main__':
    main()