import numpy as np
from pulp import *

epsilon = 1e-10

def getPolicyFromValusStar(numStates, numActions, transitions, rewards, gamma, valueStar):
    optimalPolicy = np.zeros(numStates, dtype=int)

    for state in range(numStates):
        # get the q value for all actions
        maxQValueForAction = float('-inf')
        actionToTake = -1
        for action in range(numActions):
            QValueForAction = 0
            for statePrime in range(numStates):
                QValueForAction += transitions[state, action, statePrime] * (rewards[state, action, statePrime] + gamma * valueStar[statePrime])
            if QValueForAction > maxQValueForAction:
                maxQValueForAction = QValueForAction
                actionToTake = action
        
        optimalPolicy[state] = actionToTake
            
    
    return optimalPolicy



def linearProgrammingSolver(numStates, numActions, end_states, transitions, rewards, gamma):
    Values = [LpVariable(f"V({i})") for i in range(numStates)]
    # MDPPlanningProblem = LpProblem("MDPPlanningProblem", LpMinimize)
    MDPPlanningProblem = LpProblem("MDPPlanningProblem", LpMaximize)

    # objective is maximize - sum Vs 
    
    # MDPPlanningProblem += -1 * sum(Values)
    MDPPlanningProblem += -1 * lpSum(Values)

    for state in range(numStates):
        if state in end_states:
            MDPPlanningProblem += Values[state] == 0
    

    # create equations to solve
    # using the equations we create V(s) <= sum_over_states(T(s, a , s')*(reward(s,a,si) + gamma*V(s')))

    for state in range(numStates):
        for action in range(numActions):
            intermediateValues = []
            for statePrime in range(numStates):
                intermediateValue = transitions[state, action, statePrime] * (rewards[state, action, statePrime] + gamma * Values[statePrime])
                intermediateValues.append(intermediateValue)
            # Add the constraint for every action
            # MDPPlanningProblem += Values[state] >= sum(intermediateValues)
            MDPPlanningProblem += Values[state] >= lpSum(intermediateValues)
    
    # solving the problem
        
    problemStatus = MDPPlanningProblem.solve(PULP_CBC_CMD(msg=0))

    valueStar = [Values[i].value() for i in range(numStates)]

    optimalPolicy = getPolicyFromValusStar(numStates, numActions, transitions, rewards, gamma, valueStar)

    return valueStar, optimalPolicy




def printLinearProgrammingOptimalValues(numStates, numActions, end_states, transitions, rewards, gamma):

    valueStar, optimalPolicy = linearProgrammingSolver(numStates, numActions, end_states, transitions, rewards, gamma)

    for value, action in zip(valueStar, optimalPolicy):
        print(value, action)
    
