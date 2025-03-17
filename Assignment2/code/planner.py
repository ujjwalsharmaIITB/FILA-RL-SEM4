import argparse
import numpy as np

from howardsPolicyIteration import printHowardsPolicyOptimalValues, printHowardsPolicyValue
from linearProgrammingSolver import printLinearProgrammingOptimalValues




def read_mdp_file(file_path, debug=False):
    mdp_file_contents = [x.strip() for x in open(file_path).readlines()]
    numStates = int(mdp_file_contents[0].split()[1])
    numActions = int(mdp_file_contents[1].split()[1])
    end_states = [int(x) for x in mdp_file_contents[2].split()[1:]]
    transitions = np.zeros((numStates, numActions, numStates), dtype=float)
    rewards = np.zeros((numStates, numActions, numStates), dtype=float)
    mdpType = ""
    gamma = 0.0
    for line in mdp_file_contents[3:]:
        line = line.split()
        if line[0] == "transition":
            state = int(line[1])
            action = int(line[2])
            next_state = int(line[3])
            reward = float(line[4])
            probability = float(line[5])
            transitions[state][action][next_state] = probability
            rewards[state][action][next_state] = reward
        else:
            if line[0] == "mdptype":
                mdpType = line[1]
            elif line[0] == "discount":
                gamma = float(line[1])
    
    if debug:
        print("transitions:", transitions)
        print("rewards:", rewards)
        print("numStates:", numStates)
        print("numActions:", numActions)
        print("end_states:", end_states)
        print("mdpType:", mdpType)

    return numStates, numActions, end_states, transitions, rewards, mdpType, gamma

def read_policy_file(file_path):
    policy = np.array([int(x.strip()) for x in open(file_path).readlines()], dtype=int)
    return policy






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDP Planner")
    parser.add_argument("--mdp", help="MDP file", dest="mdp", required=True)
    parser.add_argument("--algorithm", help="Algorithm to solve MDP", dest="algorithm", choices=["hpi", "lp"], default="lp")
    parser.add_argument("--policy", help="Policy file", dest="policy")
    parser.add_argument("--debug", help="Debug mode", dest="debug", action="store_true")


    args = parser.parse_args()

    mdp_file = args.mdp
    algorithm = args.algorithm
    policy_file = args.policy
    debug = args.debug



    numStates, numActions, end_states, transitions, rewards, mdpType, gamma = read_mdp_file(mdp_file, debug)

    # print(policy_file)
    if policy_file:
        policy = read_policy_file(policy_file)
        # print(policy)

    if not policy_file:
        # initialize random policy
        policy = np.random.randint(0, numActions, numStates)
    
    
    if algorithm == "hpi":
        if policy_file:
            printHowardsPolicyValue(policy, numStates, numActions, end_states, transitions, rewards, gamma)
        else:
            printHowardsPolicyOptimalValues(policy, numStates, numActions, end_states, transitions, rewards, gamma)
    elif algorithm == "lp":
        if policy_file:
            printHowardsPolicyValue(policy, numStates, numActions, end_states, transitions, rewards, gamma)
        else:
            printLinearProgrammingOptimalValues(numStates, numActions, end_states, transitions, rewards, gamma)
    else:
        print("Invalid algorithm")
        exit(0)
    



