import gym
import numpy as np
from ddpg_tf2 import Agent
from utils import plotLearning
import mujoco_py
import random
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = gym.make('FetchPush-v1')
    agent = Agent(input_dims=env.observation_space['observation'].shape, env=env,
            n_goals =env.observation_space['desired_goal'].shape,
            n_actions=env.action_space.shape[0])
    n_episodes = 5000
    
    figure_file = '/Users/admin/Documents/codes_python/rl_exam/FetchPush_her/plots/fetchpush.png'
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    success = 0
    episodes = []
    win_percent = []
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            env_dict=env.reset()
            state = env_dict['observation']
            desired_goal = env_dict['desired_goal']
            achieved_goal = env_dict['achieved_goal']
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            observation_ = observation_['observation']
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for episode in range(n_episodes):
        #observation['observation'] = env.reset()
        env_dict = env.reset()
        state = env_dict['observation']
        goal = env_dict['desired_goal']
        done = False
        transition = []
        score = 0
        for p in range(45):
            if not done:
                action = agent.choose_action(state,goal,evaluate)
                env_dict_, reward, done, info = env.step(action)
                state_ = env_dict_['observation']
                score += reward
                agent.remember(state, action, reward, state_, done, goal)
                transition.append((state, action, reward, state_))
                if not load_checkpoint:
                    agent.learn()
                state = state_
                if reward==0:
                    done = True
                    success += 1
                    print('episode finished after', p, 'steps' )
                    break
        print('success is ', success)
        if not done:
            
            new_goal = []
            for i in range(3,6):   
                new_goal.append(state[i])
            new_goal = np.array(new_goal)
            for _ in range(16):
                info = None
                state_transition = transition[_][3]
                achieved_goal = []
                for a in range(3,6):   
                    achieved_goal.append(state_transition[a])
                achieved_goal = np.array(achieved_goal)
                desired_goal = new_goal
                if not env.compute_reward(desired_goal, achieved_goal, info) == -1.0:
                     
                     agent.remember(transition[_][0], transition[_][1], transition[_][2],
                                           transition[_][3], True, new_goal)
                     agent.learn()
                     
                     break
            
            agent.remember(transition[_][0], transition[_][1], transition[_][2],
                                           transition[_][3], False, new_goal)
            agent.learn()
    
        if episode > 0 and episode % 100 == 0:
            print('success rate for last 100 episodes after', episode, ':', success)
            if len(win_percent) > 0 and (success / 100) > win_percent[len(win_percent) - 1]:
                agent.save_models()
                print('saving')
            episodes.append(episode)
            win_percent.append(success / 100)
            success = 0

    print('Episodes:', episodes)
    print('Win percentage:', win_percent)

    print('Episodes:', episodes)
    print('Win percentage:', win_percent)

    
    plt.plot(episodes, win_percent)

    plt.title('DDPG with HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Episodes')
    plt.ylim([0, 1])

    plt.savefig(figure_file)





