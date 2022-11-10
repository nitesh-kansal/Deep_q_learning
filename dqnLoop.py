import random
import torch
import numpy as np
from unityagents import UnityEnvironment
from collections import deque
from dqnagent import Agent
def convert_state(state, state_type):
    if state_type == "normal":
        state = state[None,:]
    else:
        state = np.transpose(state, (0, 3, 1, 2))
    return state

def banana_collector_dqn_training_loop(params):
    
    """Deep Q-Learning.
    
    params: dict
    ==================
        n_episodes (int) : maximum number of training episodes
        max_t (int) : maximum number of timesteps per episode
        eps_start (float) : starting value of epsilon, for epsilon-greedy action selection
        eps_end (float) : minimum value of epsilon
        eps_decay (float) : multiplicative factor (per episode) for decreasing epsilon
        
        Parameters for controlling dependence of sampling probability on TD-error in Prioritized Experience Replay
        alphaf (float) : Final value of alpha parameter in Prioritized Experience Replay 
        alpha0 (float) : Initial value of alpha parameter in Prioritized Experience Replay
        nsteps_alpha (int) : Number of episodes in which to linearly change alpha from alpha0 to alphaf
        
        Parameters for controlling Importance Sampling weight (ISw) in Prioritized Experience Replay
        betaf (float) : Final value of beta parameter in Prioritized Experience Replay 
        beta0 (float) : Initial value of beta parameter in Prioritized Experience Replay
        nsteps_beta (int) : Number of episodes in which to linearly change beta from beta0 to betaf
        
        DQN Update Parameters
        LR (float) : Learning Rate for update of DQN weights
        BUFFER_SIZE (int) : Size of the Replay Buffer
        TAU (float) : Fraction of primary network weights to be copied over to the target network after each parameter update step
                θ_target = τ*θ_primary + (1 - τ)*θ_target
        BATCH_SIZE (int) : Size of the sample to be selected at random from the Replay Buffer at each update step 
        UPDATE_EVERY (int) : Number of actions (or transitions to be recorded) to be taken before making any update to DQN weights
        SAMPLE_FREQ (int) : Number of batch sampling and DQN weight update steps to be carried out during the update step
        GAMMA (float) : Discount Factor
        IS_DDQN (bool) : Whether to enable the Double DQN improvement or continue with basic DQN
        MEMORY_TYPE ("prioritized","normal"): Whether to go with prioritized memory buffer or uniform
        IS_DUELING (bool) : Whether to enable Dueling Network improvement or not
        
        Choice of State
        STATE_TYPE ("raw_pixels","normal") : Whether to train the Agent from raw pixel states or processed state vectors
        ENVIRONMENT_PATH (string) : Complete path to the Banana Collector Environment file
        
        seed : random seed
    """
    
    # create the environment
    env = UnityEnvironment(file_name=params["ENVIRONMENT_PATH"])
    
    # Extracting the primary acting brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = convert_state(env_info.vector_observations[0] , params["STATE_TYPE"])
    state_size = state.shape
    print('States have shape:', state.shape)
    
    if params["STATE_TYPE"] == "normal":
        agent = Agent(37, 4, params["seed"], params)
    else:
        agent = Agent(3, 4, params["seed"], params)
    
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    avg_i = []
    eps = params["eps_start"]                    # initialize epsilon
    for i_episode in range(1, params["n_episodes"]+1):
        
        # alpha value
        malpha = (params["alphaf"] - params["alpha0"])/(params["nsteps_alpha"] - 1)
        alpha = malpha*(i_episode - 1) + params["alpha0"]
        
        # beta value
        mbeta = (params["betaf"] - params["beta0"])/(params["nsteps_beta"] - 1)
        beta = mbeta*(i_episode - 1) + params["beta0"]
        
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment        
        state = convert_state(env_info.vector_observations[0] , params["STATE_TYPE"])
            
        score = 0                                          # initialize the score        
        i = 0
        while i < params["max_t"]:
            action = agent.act(state, eps)       # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = convert_state(env_info.vector_observations[0] , params["STATE_TYPE"])
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done, alpha = alpha, beta = beta)
            score += reward                                # update the score
            state = next_state  # roll over the state to next time step
            i += 1
            if done:                                       # exit loop if episode finished
                break 
        avg_i.append(i)
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(params["eps_end"], params["eps_decay"]*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Episode Len : {}'.format(i_episode, np.mean(scores_window), np.mean(avg_i)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Episode Len : {}'.format(i_episode, np.mean(scores_window), np.mean(avg_i)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            print('Saving DQN weights in ./checkpoint.pth')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    print("******************* watch trained agent in action **********************")
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = convert_state(env_info.vector_observations[0] , params["STATE_TYPE"])        # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, eps=0)          # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = convert_state(env_info.vector_observations[0] , params["STATE_TYPE"])   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break

    print("Score of the sample episode after training: {}".format(score))
    env.close()
    return scores