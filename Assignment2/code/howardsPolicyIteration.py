import numpy as np

epsilon = 1e-10

def getOptimalValuesUsingValueIteration(numStates, numActions, end_states, transitions, rewards, gamma):
    # initialize V(s) for all states
    Value = np.zeros(numStates, dtype=float)
    # epsilon = 1e-10
    # loop until V converges
    while True:
        Value_New = np.zeros(numStates, dtype=float)
        for state in range(numStates):
            if state in end_states:
                Value_New[state] = 0
            else:
                # calculate max_a sum_s' P(s' | s, a) * (R(s, a, s') + gamma * V(s'))
                for action in range(numActions):
                    for statePrime in range(numStates):
                        newStateValue = transitions[state][action][statePrime] * (rewards[state][action][statePrime] + gamma * Value[statePrime])
                        
        
                        Value_New[state] = max(Value_New[state], newStateValue)
        Value = Value_New
        # check for convergence
        if np.max(np.abs(Value_New - Value)) < epsilon:
            break
    return Value
            

def getQValue(numStates, state, action, transitions, rewards, Value, gamma):
    qValue = 0
    for statePrime in range(numStates):
        qValue += transitions[state][action][statePrime] * (rewards[state][action][statePrime] + gamma * Value[statePrime])
    return qValue



def valueIterationSolver(policy, numStates, numActions, end_states, transitions, rewards, gamma):
    # definition from RL book page 74
    # initialize V(s) for all states
    Value = np.zeros(numStates, dtype=float)
    # epsilon = 1e-4
    # loop until V converges
    i = 0
    while True:
        i+=1
        Value_New = np.zeros(numStates, dtype=float)
        for state in range(numStates):
            if state in end_states:
                Value_New[state] = 0
            else:
                # do policy iteration
                # policy is statiorary and deterministic so evaluation is 
                action = policy[state]
                value_prime = 0
                for statePrime in range(numStates):
                    newStateValue = transitions[state][action][statePrime] * (rewards[state][action][statePrime] + gamma * Value[statePrime])
                    value_prime +=  newStateValue
                Value_New[state] = value_prime
        # check for convergence
        if np.max(np.abs(Value_New - Value)) < epsilon:
            break
        Value = Value_New
    # print("Value Iteration took", i, "iterations")

    return Value




    

def howardsPolicyIteration(policy, numStates, numActions, end_states, transitions, rewards, gamma):
    # get the optimal values using value iteration

    
    # initialize policy_stable to false
    policy_stable = False
    ValueForPolicy = valueIterationSolver(policy, numStates, numActions, end_states, transitions, rewards, gamma)
    QValuesForPolicy = np.zeros((numStates), dtype=float)
    UpdatedPolicy = policy.copy()

    for state, action in enumerate(UpdatedPolicy):
        QValuesForPolicy[state] = getQValue(numStates, state, action, transitions, rewards, ValueForPolicy, gamma)
        
    # loop until policy_stable is true
    while not policy_stable:
        # policy evaluation
        ValueForCurrentPolicy = valueIterationSolver(UpdatedPolicy, numStates, numActions, end_states, transitions, rewards, gamma)
    
        improvedStates = {}
        for state in range(numStates):
            improvedStates[state] = []

        for state in range(numStates):
            for action in range(numActions):
                QValueForAction = getQValue(numStates, state, action, transitions, rewards, ValueForCurrentPolicy, gamma)
                if QValueForAction > QValuesForPolicy[state]:
                    # append action and Qvalue
                    improvedStates[state].append((action, QValueForAction))
        
        # if no improvement in policy, break
        stable = True
        for state in range(numStates):
            stable = stable and (len(improvedStates[state]) == 0)
        if stable:
            policy_stable = True
            break
        
        # policy improvement
        for state in range(numStates):
            if len(improvedStates[state]) > 0:
                # sort the actions based on QValue
                maxAction = max(improvedStates[state], key=lambda x: x[1])
                UpdatedPolicy[state] = maxAction[0]
                QValuesForPolicy[state] = maxAction[1]
        
    return UpdatedPolicy, QValuesForPolicy


def printHowardsPolicyOptimalValues(policy, numStates, numActions, end_states, transitions, rewards, gamma):
    policy, QValues = howardsPolicyIteration(policy, numStates, numActions, end_states, transitions, rewards, gamma)
    for state in range(numStates):
        print(QValues[state], policy[state])



def printHowardsPolicyValue(policy, numStates, numActions, end_states, transitions, rewards, gamma):
    # print("Calling")
    Value = valueIterationSolver(policy, numStates, numActions, end_states, transitions, rewards, gamma)
    for state in range(numStates):
        print(Value[state], policy[state])

