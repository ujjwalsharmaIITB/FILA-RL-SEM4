# Task 3
# Using inspiration from code in task1.py and simulator.py write the appropriate functions to create the plot required.

import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import *
from task1 import Algorithm
from multiprocessing import Pool
import time


# DEFINE your algorithm class here

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
    def __init__(self, epsilon, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = epsilon
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



# DEFINE single_sim_task3() HERE


def single_sim_task3(seed, epsilon, PROBS=[0.7, 0.6, 0.5, 0.4, 0.3], HORIZON=30000):
  np.random.seed(seed)
  shuffled_probs = np.random.permutation(PROBS)
  bandit = BernoulliBandit(probs=shuffled_probs)
  algo_inst = Eps_Greedy(epsilon, num_arms=len(shuffled_probs), horizon=HORIZON)

#   pull each arm at least once
  for i in range(len(shuffled_probs)):
      reward = bandit.pull(i)
      algo_inst.get_reward(arm_index=i, reward=reward)

  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.regret()

# DEFINE simulate_task3() HERE





def simulate_task3(epsilon, num_sims=50):
    def multiple_sims(epsilon, num_sims=50):
        with Pool(10) as pool:
            sim_out = pool.starmap(single_sim_task3, [(i, epsilon) for i in range(num_sims)])
        return sim_out 
    
    sim_out = multiple_sims(epsilon, num_sims)
    regrets = np.mean(sim_out)

    return regrets

# DEFINE task3() HERE

def task3(num_sims=50):
    """generates the plots and regrets for task1
    """
    regrets = []
    start_time = time.time()
    remaining_loops = 100
    minimum_regret = float('inf')
    min_regret_epsilon = 0
    for epsilon in np.arange(0, 1.01, 0.01):
        regret_at_epsilon = simulate_task3(epsilon, num_sims)
        regrets.append(regret_at_epsilon)
        if regret_at_epsilon < minimum_regret:
            minimum_regret = regret_at_epsilon
            min_regret_epsilon = epsilon
        check_time = time.time()
        remaining_time = (check_time - start_time) * remaining_loops / (100 - remaining_loops + 1)
        print(f"Time Spent: {check_time - start_time} sec, Time remaining {remaining_time} secs.")
        remaining_loops -= 1

    print("Regrets: ", regrets)
    print("Minimum Regret: ", minimum_regret)
    print("Epsilon for Minimum Regret: ", min_regret_epsilon)
    plt.plot(np.arange(0, 1.01, 0.01), regrets)
    plt.title("Regret vs Epsilon")
    # plt.xticks(np.arange(0, 1.01, 0.05))
    plt.savefig("task3-pulled-once-{}.png".format(time.strftime("%Y%m%d-%H%M%S")))
    plt.clf()



# Call task3() to generate the plots
  
# task3()
  
# print(single_sim_task3(0.1)) # 0.1
  
if __name__ == "__main__":
    task3()