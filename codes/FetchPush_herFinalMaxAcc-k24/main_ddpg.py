import gym
import numpy as np
from ddpg_tf2 import Agent
import mujoco_py
import random
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    env = gym.make('FetchPush-v1')
    agent = Agent(input_dims=env.observation_space['observation'].shape, env=env,
            n_goals =env.observation_space['desired_goal'].shape,
            n_actions=env.action_space.shape[0])
    n_episodes = 16
    n_epochs = 200
    n_cycles = 50
    figure_file = './plots/fetchpushP3.png'
    
    
    load_checkpoint = True
    success = 0
    epochs = []
    win_percent = []
    win_percent_test=0
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
        #agent.load_models()
        evaluate = True
    else:
        evaluate = False
    if load_checkpoint:
        agent.load_models()

        for episode in range(100):
                env_dict = env.reset()
                state = env_dict['observation']
                goal = env_dict['desired_goal']
                done=False
                
                for p in range(49):
                    if not done:
                     
                        action = agent.choose_action(state,goal,evaluate)
                        env_dict_, reward, done, info = env.step(action)
                        state_ = env_dict_['observation']
                        
                        env.render()
                        
                        if reward ==0:
                            done = True
                        state = state_
                        if done:
                            success += 1
                            print('episode finished after steps                                ', p )
                            break
                
                win_percent_test += success
                print(win_percent_test)
            


    if not load_checkpoint:        
        
        for epoch in range(n_epochs):

            
            for cycle in range(n_cycles):
                
                for episode in range(n_episodes):
                    env_dict = env.reset()
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
                            #env.render()
                            for x in range(3,6):
                                achieved_goal1.append(state_[x])
                            achieved_goal1 = np.array(achieved_goal1)
                            if not load_checkpoint:
                                agent.remember(state, action, reward, state_, done, goal)
                                transition.append((state, action, reward, state_, goal, achieved_goal1))
                            if reward ==0:
                                done = True
                            state = state_
                            if done:
                                success += 1
                                print('episode finished after steps                                ', p )
                                break
                    if not done:
                        if not load_checkpoint:
                            #we want the distance to be greater than 0, since it is the distance between initial and final,
                                # that is 0 for most cases.
                                for _ in range(24): #sampling strategy is "episodic"  
                                    index = random.randint(0,48)
                                    box_rand = np.copy(transition[index][5])
                                    if  np.linalg.norm(box_rand - goal, axis=-1)+0.005 < np.linalg.norm(goal - initial_box, axis=-1):
                                        #this condition needs to be modified, it doesn't make that much sense... even if it works!!
                                         print('                                                                                        yeah!!')
                                         agent.remember(transition[index][0], transition[index][1], 0,
                                                               transition[index][3], True, box_rand)
                                         
                                
                                    agent.remember(transition[index][0], transition[index][1], transition[index][2],
                                                               transition[index][3], False, box_rand)
                #at the end of each cycle are performed 40 optimization steps:
                for _ in range(40):
                    agent.learn()
                print('succes after the ', cycle, 'cycle is ', success) 

            print('succes after the ', epoch, 'epoch is ', success)        

            if not load_checkpoint:
                if len(win_percent) > 0 and (success / 800) > max(win_percent):
                    agent.save_models()
                    print('saving')
                epochs.append(epoch)
                win_percent.append(success / 800)
                success = 0

    
    if not load_checkpoint:
        
        pickle.dump(win_percent, open("./winpercentk24P3.txt", "wb"))

        print("win_percent from file: ")

        win = pickle.load(open("./winpercentk24P3.txt", "rb"))

        for i in range(len(win)):
            
            print(win[i])
        
        print(epochs)
        print(win_percent)

        
        plt.plot(epochs, win_percent)

        plt.title('DDPG with HER')
        plt.ylabel('Win Percentage')
        plt.xlabel('Number of Epochs')
        plt.ylim([0, 1])
        

        plt.savefig(figure_file)
        plt.show()





