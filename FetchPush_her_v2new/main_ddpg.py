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
    n_episodes = 16
    n_epochs = 200
    n_cycles = 50
    figure_file = '/Users/admin/Documents/codes_python/rl_exam/FetchPush_her_v2/plots/fetchpush.png'
    
    
    load_checkpoint = True
    success = 0
    epochs = []
    win_percent = []
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            env_dict=env.reset()
            state = env_dict['observation']
            goal = env_dict['desired_goal']
            action = env.action_space.sample()
            env_dict_, reward, done, info = env.step(action)
            state_ = env_dict_['observation']
            agent.remember(state, action, reward, state_, done, goal)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
    if load_checkpoint:
        agent.load_models()
    for epoch in range(n_epochs):
        
        for cycle in range(n_cycles):
            
            for episode in range(n_episodes):
                #observation['observation'] = env.reset()
                env_dict = env.reset()
                #action = env.action_space.sample()
                #env_dict, reward, done, info = env.step(action)
                state = env_dict['observation']
                goal = env_dict['desired_goal']
                done=False
                initial_box =[]
                for l in range(3,6):
                    initial_box.append(state[l])
                initial_box = np.array(initial_box)
                transition = []
                score = 0
                for p in range(49):
                    if not done:
                     
                        if random.randint(0,9) > 1: #20 percent of prob that it will be performed a random action
                            action = agent.choose_action(state,goal,evaluate)
                        else:
                            action = env.action_space.sample()
                        env_dict_, reward, done, info = env.step(action)
                        state_ = env_dict_['observation']
                        achieved_goal1 = []
                        for x in range(3,6):
                            achieved_goal1.append(state_[x])
                        achieved_goal1 = np.array(achieved_goal1)
                        if not load_checkpoint:
                            agent.remember(state, action, reward, state_, done, goal)
                            transition.append((state, action, reward, state_, goal, achieved_goal1))
                            #agent.learn()
                        if reward ==0:
                            done = True
                        state = state_
                        if done:
                            #final_box_pos =[]
                            #for o in range(3,6):
                                #final_box_pos.append(state[o])
                            #final_box_pos = np.array(final_box_pos)
                            #if  np.linalg.norm(final_box_pos - initial_box, axis=-1)>=0.005
                            success += 1
                            print('episode finished after steps                                ', p )
                            break
                if not done:
                    if not load_checkpoint:
                        
                        #print('success is ', success, 'after', episode, 'episodes')
                        #final_box_pos =[] 
                        #for l in range(3,6):
                         #   final_box_pos.append(state[l])
                        #final_box_pos = np.array(final_box_pos)
                        #if np.linalg.norm(final_box_pos - initial_box, axis=-1)>=0.005: ## entering her #
                            #we want the distance to be greater than 0, since it is the distance between initial and final,
                            # that is 0 for most cases.
                            for _ in range(8): #sampling strategy is "episodic"  
                                index = random.randint(0,48)
                                box_rand = np.copy(transition[index][5])
                                #end_eff_pos = transition[_][3][:3]
                                if  np.linalg.norm(box_rand - goal, axis=-1) <= np.linalg.norm(goal - initial_box, axis=-1):
                                    #this condition needs to be modified, it doesn't make that much sense... even if it works!!
                                     print('                                                                                        yeah!!')
                                     agent.remember(transition[index][0], transition[index][1], 0,
                                                           transition[index][3], True, box_rand)
                                     #agent.learn()
                                     
                            
                                agent.remember(transition[index][0], transition[index][1], transition[index][2],
                                                           transition[index][3], False, box_rand)
                                #agent.learn()
            #at the end of each cycle are performed 40 optimization steps:
            for _ in range(40):
                agent.learn()
            print('succes after the ', cycle, 'cycle is ', success) 
                

        if not load_checkpoint:
            #if episode > 0 and episode % 100 == 0:
                #print('success rate for last 100 episodes after', episode, ':', success)
            if len(win_percent) > 0 and (success / 50*16) > max(win_percent):
                agent.save_models()
                print('saving')
            epochs.append(epoch)
            win_percent.append(success / 50*16)
            success = 0
    if not load_checkpoint:
        


        
    plt.plot(epochs, win_percent)

    plt.title('DDPG with HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Epochs')
    plt.ylim([0, 1])

    plt.savefig(figure_file)




