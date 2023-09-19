import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque, namedtuple
import copy

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity: int
            max length of buffer deque

        Returns
        -------
        None
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self,
             state: list,
             action: list,
             reward: list,
             next_state: list,
             mask: list):
        """
        Insert a new memory into the end of the ReplayMemory buffer

        Parameters
        ----------
        state, action, reward, next_state, mask: array_like
        Returns
        -------
        None
        """
        self.buffer.insert(0, (state, action, reward, next_state, mask))

    def sample(self, batch_size, c_k):
        N = len(self.buffer)
        if c_k>N:
            c_k = N
        indices = np.random.choice(c_k, batch_size)
        batch = [self.buffer[idx] for idx in indices]
        #batch = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, mask = map(np.stack,zip(*batch))
        return state, action, reward, next_state, mask

    def __len__(self):
        return len(self.buffer)

class HerReplayMemory(ReplayMemory):
    def __init__(self,
                 capacity: int,
                 env,
                 strategy: str='final'):
        """
        Initialize HerReplayMemory object

        Parameters
        ----------
        capacity: int
        env: AMRL.RealExpEnv
        strategy: str

        Returns
        -------
        None
        """
        super(HerReplayMemory, self).__init__(capacity)
        self.env = env
        self.n_sampled_goal = 2
        self.strategy = strategy

    def sample(self,
               batch_size: int,
               c_k: float) -> tuple:
        """
        Sample batch_size (state, action, reward, next_state, mask) # of memories
        from the HERReplayMemory, emphasizing the c_k most recent experiences
        to account for potential tip changes.

        Also implemented: hindsight experience replay, which treats
        memories in which the achieved goal was different than the intended goal
        as 'succesful' in order to speed up training.


        Parameters
        ----------
        batch_size: int
        c_k: int
            select from the c_k most recent memories

        Returns
        -------
        tuple
        """
        N = len(self.buffer)
        if c_k>N:
            c_k = N
        indices = np.random.choice(c_k, int(batch_size))
        batch = []
        for idx in indices:
            batch.append(self.buffer[idx])
            state, action, reward, next_state, mask = self.buffer[idx]
            #print('old state:', state, 'old next state:', next_state, 'old reward:', reward)

            final_idx = self.sample_goals(idx)
            for fi in final_idx:
                _, _, _, final_next_state, _ = self.buffer[fi]
                new_next_state = copy.copy(next_state)
                new_state = copy.copy(state)
                new_state[:2] = final_next_state[2:]
                new_next_state[:2] = final_next_state[2:]
                new_reward = self.env.compute_reward(new_state, new_next_state)
                m = (new_state, action, new_reward, new_next_state, mask)
                batch.append(m)
        print('No. of samples:', len(batch))
        state, action, reward, next_state, mask = map(np.stack,zip(*batch))
        return state, action, reward, next_state, mask

    def sample_goals(self, idx):
        """
        Sample memories in the same episode

        Parameters
        ----------
        idx: int

        Returns
        -------
        array_like
            list of final_idx HerReplayMemory buffer indices
        """
        #get done state idx
        i = copy.copy(idx)

        while True:
            _,_,_,_,m = self.buffer[i]
            if not m:
                break
            else:
                i-=1
        if self.strategy == 'final' or i == idx:
            return [i]
        elif self.strategy == 'future':
            iss = np.random.choice(np.arange(i, idx+1), min(idx-i+1, 3))
            return iss

    def calculate_value(self, state):
        """
        Deprecated
        """
        goal_nm = state[:2]*self.env.goal_nm
        atom_nm = state[2:]*self.env.goal_nm
        dist_destination = np.linalg.norm(atom_nm - goal_nm)
        a = atom_nm
        b = goal_nm
        cos_similarity_destination = np.inner(a,b)/(self.env.goal_nm*np.clip(np.linalg.norm(a), a_min=self.env.goal_nm, a_max=None))
        value = self.env.calculate_value(dist_destination, cos_similarity_destination)
        return value