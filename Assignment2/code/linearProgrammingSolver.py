import numpy as np
from pulp import *

epsilon = 1e-10

def getPolicyFromValusStar(numStates, numActions, transitions, rewards, gamma, valueStar):
    optimalPolicy = np.zeros(numStates, dtype=int)

    for state in transitions.keys():
        # get the q value for all actions
        maxQValueForAction = float('-inf')
        actionToTake = -1
        for action in transitions[state].keys():
            QValueForAction = 0
            for statePrime in transitions[state][action].keys():
                QValueForAction += transitions[state][action][statePrime] * (rewards[state, action, statePrime] + gamma * valueStar[statePrime])
            if QValueForAction > maxQValueForAction:
                maxQValueForAction = QValueForAction
                actionToTake = action
        
        optimalPolicy[state] = actionToTake
            
    
    return optimalPolicy



def linearProgrammingSolver(numStates, numActions, end_states, transitions, rewards, gamma):
    Values = [LpVariable(f"V({i})") for i in range(numStates)]
    # MDPPlanningProblem = LpProblem("MDPPlanningProblem", LpMinimize)
    MDPPlanningProblem = LpProblem("MDPPlanningProblem", LpMaximize) # Also tried using this 

    # objective is Maximize the negative sum Vs 
    # definition taken from the slides
    
    # MDPPlanningProblem += sum(Values)  if LpMinimize is used
    # MDPPlanningProblem += -1 * sum(Values)
    # https://coin-or.github.io/pulp/main/includeme.html#useful-functions
    MDPPlanningProblem +=  -1 * lpSum(Values)

    for state in range(numStates):
        if state in end_states:
            MDPPlanningProblem += Values[state] == 0
        if state not in transitions.keys():
            MDPPlanningProblem += Values[state] == 0
    

    # create equations to solve
    # using the equations we create V(s) <= sum_over_states(T(s, a , s')*(reward(s,a,si) + gamma*V(s')))

    for state in transitions.keys():
        for action in transitions[state].keys():
            intermediateValues = []
            for statePrime in transitions[state][action].keys():
                intermediateValue = transitions[state][action][statePrime] * (rewards[state, action, statePrime] + gamma * Values[statePrime])
                intermediateValues.append(intermediateValue)
            # Add the constraint for every action
            # MDPPlanningProblem += Values[state] >= sum(intermediateValues)
            MDPPlanningProblem += Values[state] >= lpSum(intermediateValues)
    
    # solving the problem
        
    problemStatus = MDPPlanningProblem.solve(PULP_CBC_CMD(msg=0))

    valueStar = [Values[i].value() for i in range(numStates)]

    optimalPolicy = getPolicyFromValusStar(numStates, numActions, transitions, rewards, gamma, valueStar)

    return valueStar, optimalPolicy



def getNewTransitionMatrix(numStates, numActions, transitions):
    # create a dense matrix by removing all the zeros
    newTransitionMatrix = {}
    for state in range(numStates):
        for action in range(numActions):
            for statePrime in range(numStates):
                if transitions[state, action, statePrime] > 0:
                    if state not in newTransitionMatrix:
                        newTransitionMatrix[state] = {}
                    if action not in newTransitionMatrix[state]:
                        newTransitionMatrix[state][action] = {}
                    newTransitionMatrix[state][action][statePrime] = transitions[state, action, statePrime]
    return newTransitionMatrix



def printLinearProgrammingOptimalValues(numStates, numActions, end_states, transitions, rewards, gamma):
    newTransitionMatrix = getNewTransitionMatrix(numStates, numActions, transitions)
    valueStar, optimalPolicy = linearProgrammingSolver(numStates, numActions, end_states, newTransitionMatrix, rewards, gamma)

    for value, action in zip(valueStar, optimalPolicy):
        print(value, action)
    
