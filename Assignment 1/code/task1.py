"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        # self.ucb = np.zeros(num_arms)
        # initially i tried with zeros but in that case only one of the arms was getting pulled
        # so the solution was to initialize the ucb with ones
        self.ucb = np.ones(num_arms)
        
        # counts will be used to store the number of times each arm has been pulled
        self.counts = np.ones(num_arms)
        
        # values will be used to store the empirical mean of each arm
        self.values = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # give the arrm with the highest UCB
        # index_to_pull = np.argmax(self.ucb)
        # print("Pulled" , index_to_pull)
        return np.argmax(self.ucb)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # here we update the empirical mean of the arm and the UCB of the arm
        # arm_index is the arm that was pulled
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        # get the updated emperical reward of the arm, (I am using the same formula as in the Eps_Greedy algo.)
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        # update the UCB of the arm
        # it was mentioned that math.log is faster than np.log for scalars so i am using math.log
        
        ucb_term_for_arm = math.sqrt((2* math.log(self.horizon)) / n)
        
        self.ucb[arm_index] = self.values[arm_index] + ucb_term_for_arm
        # print("Updated UCB", self.ucb)

        # END EDITING HERE


def kl_divergence(p, q):
    # This definition of KL divergence is taken from the slides Lecture 2, slide 55 and
    # https://github.com/guptav96/bandit-algorithms/blob/main/algo/klucb.py
    
    max_value_for_divergence = float('inf')

    if q == 0 and p == 0:
        return 0
    elif q == 0 and not p == 0:
        return max_value_for_divergence
    elif q == 1 and p == 1:
        return 0
    elif q == 1 and not p == 1:
        return max_value_for_divergence
    elif p == 0:
        return math.log(1 / (1 - q))
    elif p == 1:
        return math.log(1 / q)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))




#  From the slides, Lecture 3, slide 12:
#  ucb-klt a is the solution q ∈ [pta, 1] to KL(pta, q) = ln(t)+c*ln(ln(t))/uta.
# We have to define KL divergence

# Also in the slides it is given that KL(p,q) increases linearly with p and q so we can use
# binary search to find the value of q that satisfies the equation

# uta*KL(pta, q) ≤ ln(t) + c ln(ln(t))

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        # counts initialized by 1
        
        self.counts = np.ones(num_arms)
        # values is for empirical mean
        self.values = np.zeros(num_arms)
        self.c  = 3
        self.contanst_value = math.log(horizon) + self.c * math.log(math.log(horizon))
        self.ucbs = np.zeros(num_arms)
        # END EDITING HERE

    
    def give_pull(self):
        # START EDITING HERE
        # calculate ucb value for each arm
        for i in range(self.num_arms):
            max_iters = 1000 # maximum number of iterations, reference taken from https://github.com/guptav96/bandit-algorithms/blob/main/algo/klucb.py
            lower = self.values[i]
            upper = 1
            iters = 0
            precision_value = 1e-4
            q = (lower + upper) / 2
            while iters < max_iters and (upper - lower) > precision_value:
                iters += 1
                q = (lower + upper) / 2
                uta = self.counts[i]
                pta = self.values[i]
                if uta * kl_divergence(pta, q) <= self.contanst_value: # using this condition the loop was taking less time. took 160.04 seconds
                    lower = q + precision_value
                else:
                    upper = q - precision_value
            
            # kl(pta,q) is greater than the constant value, hence ucb should be lower value
            self.ucbs[i] = lower

        return np.argmax(self.ucbs)
        
        # END EDITING HERE
    


    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # increment the counts of the arm
        self.counts[arm_index] += 1
        n = self.counts[arm_index]

        # update the empirical mean of the arm
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
        # Beta(sa + 1, fa + 1) is the posterior distribution of the success probability of an arm
        
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # select the arm with maximum sample from the posterior distribution
        
        # This code takes time, np.beta also takes individual items
        # sample_values = np.random.beta(self.successes + 1, self.failures + 1)
        # return np.argmax(sample_values)
    
        max_sample_value = -1
        max_sample_index = -1
        for i in range(self.num_arms):
            sample = np.random.beta(self.successes[i] + 1, self.failures[i] + 1)
            if sample > max_sample_value:
                max_sample_value = sample
                max_sample_index = i
        return max_sample_index
        

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # update the successes and failures of the arm
        # arm_index is the arm that was pulled

        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1
        # END EDITING HERE

