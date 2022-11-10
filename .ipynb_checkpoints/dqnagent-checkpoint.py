import numpy as np
import random
from collections import namedtuple, deque
from copy import deepcopy

from qmodel import QNetwork, QNetworkPixel

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, params):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if params["STATE_TYPE"] == "raw_pixels":
            self.qnetwork_local = QNetworkPixel(state_size, action_size, seed, params["IS_DUELING"]).to(device)
            self.qnetwork_target = QNetworkPixel(state_size, action_size, seed, params["IS_DUELING"]).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed, params["IS_DUELING"]).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed, params["IS_DUELING"]).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=params["LR"])

        # Replay memory
        if params["MEMORY_TYPE"] == "prioritized":
            self.memory = PrioritizedReplayBuffer(action_size, params["BUFFER_SIZE"], params["BATCH_SIZE"], seed)
        else:
            self.memory = ReplayBuffer(action_size, params["BUFFER_SIZE"], params["BATCH_SIZE"], seed)
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, alpha=0., beta=0.):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.params["UPDATE_EVERY"]
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.length() > self.params["BATCH_SIZE"]:
                self.learn(self.params["GAMMA"],self.params["SAMPLE_FREQ"], alpha, beta)

    def act_value_proportional(self, state, eps=0.):
        """Returns actions for given state as per value proportional policy.
            i.e. 
            
            q(a,s) =  (Q(a,s) - min a' [Q(a',s)] + e)
            
            p(a/s) = ((q(a/s)/sum[q(a/s)]) * epsilon) + 1. - epsilon       a = argmax a' [Q(a,s)]
                   = ((q(a/s)/sum[q(a/s)]) * epsilon)                      otherwise
                   
            I saw this policy perform significantly better for taxi-v2 and taxi-v3.
            
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Better Epsilon-Greedy action selection, may be epsilon needs to be higher
        action_values = action_values.cpu().data.numpy()[0,:]
        policy = (action_values - action_values.min() + 1e-6)
        policy = np.log(1.+ policy)
        policy = eps * policy/policy.sum()
        policy[np.argmax(action_values)] += 1. - eps
        return np.random.choice(np.arange(self.action_size), p = policy)

    def act(self, state, eps=0.):
        """Returns actions for given state as per normal epsilon-greedy policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, gamma, sample_freq, alpha, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float) : discount factor
            alpha (float) : power in prioritization
            beta (float)  : power in importance sampling using for prioritization
        """
        
        for i in range(sample_freq):
            experiences = self.memory.sample(alpha, beta)
            states, actions, rewards, next_states, dones, ids, ISw = experiences

            # implemented Double DQN
            self.qnetwork_target.eval()
            self.qnetwork_local.eval()
            with torch.no_grad():
                if self.params["IS_DDQN"]:
                    best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
                    q_target = self.qnetwork_target(next_states).gather(1, best_actions)
                else:
                    q_target = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
                q_target = rewards + gamma*q_target*(1.-dones)
                
            self.qnetwork_local.train()
            q_pred = self.qnetwork_local(states).gather(1, actions)
            loss = (ISw * (q_pred - q_target) ** 2).mean()
                        
            # ********************************* PRIORITIZED REPLAY **************************#
            if self.params["MEMORY_TYPE"] == "prioritized":
                # updating priority of a sample
                new_scores = torch.abs(q_pred - q_target).squeeze(1).detach().cpu().data.numpy() + 1e-3
                self.memory.update(ids,new_scores)
            # ********************************* ****************** **************************#
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.params["TAU"]) 
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size 
        self.buffer_size = buffer_size
        self.memory = [0]*buffer_size
        self.scores = np.zeros(shape=buffer_size)
        self.len = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
                
                
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""        
        e = self.experience(state, action, reward, next_state, done)
        l = min([self.len, self.buffer_size])
        if l > 0:
            score = self.scores[:l].max()
        else:
            score = 2.
        self.memory[self.len % self.buffer_size] = e
        self.scores[self.len % self.buffer_size] = score
        self.len += 1
    
    def update(self, ids, new_scores):
        #print(ids.shape, new_scores.shape, ids[:3], self.scores[ids[0]], self.scores[ids[1]], self.scores[ids[2]])
        self.scores[ids] = new_scores
    
    def sample(self, alpha, beta):
        """Randomly sample a batch of experiences from memory."""
        l = min([self.len, self.buffer_size])
        t = np.power(self.scores[:l],alpha)
        t = t/t.sum()
        ids = np.random.default_rng().choice(l, size=self.batch_size, replace = False, p = t).astype(int)
        ISw = np.power(t[ids]*l,-beta)
        ISw = ISw/ISw.mean()
        #print(ISw.max(), ISw.min(), ISw.mean(), ISw.shape, t.shape, ids.shape, t[ids], t[ids[0]], t[ids[1]], t[ids[2]]) 
        
        ISw = torch.from_numpy(ISw).float().to(device)
        states = torch.from_numpy(np.vstack([self.memory[ind].state for ind in ids if self.memory[ind] is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[ind].action for ind in ids if self.memory[ind] is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[ind].reward for ind in ids if self.memory[ind] is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[ind].next_state for ind in ids if self.memory[ind] is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[ind].done for ind in ids if self.memory[ind] is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones, ids, ISw)
    
    def length(self):
        """Return the current size of internal memory."""
        return min([self.len, self.buffer_size])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, alpha, beta):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        ISw = torch.from_numpy(np.ones(self.batch_size)).float().to(device)
        return (states, actions, rewards, next_states, dones, [], ISw)

    def length(self):
        """Return the current size of internal memory."""
        return len(self.memory)