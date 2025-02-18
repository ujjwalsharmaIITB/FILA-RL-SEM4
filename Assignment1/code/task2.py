"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the CostlySetBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_query_set(self): This method is called when the algorithm needs to
        provide a query set to the oracle. The method should return an array of 
        arm indices that specifies the query set.
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_query_set method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
"""

import numpy as np
from task1 import Algorithm
# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

# Here the problem is that we will have to select the pulls from a set
# of arms and after selecting we would get the reward of the arm and a 
# penalty for selecting the arm. The goal is to maximize the reward.





class CostlySetBanditsAlgo(Algorithm):
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon

        self.succesful_pulls = np.zeros(num_arms)
        self.failed_pulls = np.zeros(num_arms)

        # END EDITING HERE

    
    def give_query_set(self):
        # START EDITING HERE
        # We need to make this penalty as small as possible while making sure that we get a high reward.

        # Given a set {a1, a2…, ak}, the penalty is 1/k.
        # The expected reward from this set is  E(s)= (sum(s) - 1)/ k   
 
        
        # if np.random.random() < self.epsilon:
        #     return np.arange(self.num_arms)

        beta_values = np.random.beta(self.succesful_pulls + 1, self.failed_pulls + 1)
        
        # sort the values of the samples so that we can select the best arms 
        # https://numpy.org/doc/2.1/reference/generated/numpy.argsort.html
        sorted_arms = np.argsort(beta_values)[: : -1]
        # index of the arm with the maximum value
        
        max_expected_reward = float('-inf')
        cumulative_sum = 0
        index_to_be_selected = -1

        # get a subset from these sorted sample values such that
        # the expected reward becomes maximum.
        for i in range(self.num_arms):
            # penalty_value = 1/(i + 1)
            # The reason that this will give the set with maximum expected reward
            # is that because the values are sorted in descending order
            # a particular subset constructed in this manner will contain
            # the maximum values that can come inside the subset of the same size.

            cumulative_sum = cumulative_sum + beta_values[sorted_arms[i]]
            expected_reward = (cumulative_sum - 1) / (i + 1)
            if expected_reward > max_expected_reward:
                max_expected_reward = expected_reward
                index_to_be_selected = i
            
        
        # max_min_value_index = np.argmax(sorted_sub_values)
        # print("*"*10)
        # print("Values" , self.values)
        # print("Sorted arms" , sorted_arms)
        # print("Sorted sub values" , sorted_sub_values)l
        # print(max_min_value_index)
        # print("*"*10)
        # select from the sorted arms till the max_min_value
        
        return sorted_arms[:index_to_be_selected + 1]
        
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # update the empirical mean of the arm

        if reward == 1:
            self.succesful_pulls[arm_index] += 1
        else:
            self.failed_pulls[arm_index] += 1
        