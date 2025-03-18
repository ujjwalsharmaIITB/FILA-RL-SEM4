import argparse

action2IDs = {
    "forward": 0,
    "left": 1,
    "right": 2,
    "turn": 3
}

IDs2Actions = {
    0: "forward",
    1: "left",
    2: "right",
    3: "turn"
}


def readGridWorld(gridWorldPath):
    gridWorld = [[x.strip() for x in sentence.strip().split(" ")] for sentence in open(gridWorldPath).readlines()]
    numRows = len(gridWorld)
    numCols = len(gridWorld[0])
    return numRows, numCols, gridWorld


directions2Ids = {
    "^" : 0,
    ">" : 1,
    "v" : 2,
    "<" : 3
}

ids2Directions = {
    0 : "^",
    1 : ">",
    2 : "v",
    3 : "<"
}



def actionOnDirection(direction, action):
    if action == "left":
        return (direction - 1) % 4
    elif action == "right":
        return (direction + 1) % 4
    elif action == "forward":
        return direction
    elif action == "turn":
        return (direction + 2) % 4
    

 
def generateStatesWithKeyValue(gridWorld, row, col, direction, key, validState, stateId, startDirection):
    validState[(row, col, direction, key)] = {
        "stateID": stateId
    }

    validState[(row, col, direction, key)]['goal'] = False
    validState[(row, col, direction, key)]['start'] = False
    validState[(row, col, direction, key)]['key'] = False
    validState[(row, col, direction, key)]['door'] = False
    if key == 1 and gridWorld[row][col] == 'g':
        validState[(row, col, direction, key)]['goal'] = True
    
    if gridWorld[row][col] == 's' and key == 0 and direction == startDirection:
        validState[(row, col, direction, key)]['start'] = True
    
    if gridWorld[row][col] == 'k' and key == 0:
        validState[(row, col, direction, key)]['key'] = True
    
    if gridWorld[row][col] == 'd' and key == 1:
        validState[(row, col, direction, key)]['door'] = True
    
    return validState

   

def generateStatesFromGridWorld(numRows, numCols, gridWorld):
    validStates = {}
    stateId = -1
    isKeyAvaible = False
    startDirection = "<"
    for row in range(numRows):
        for col in range(numCols):
            if gridWorld[row][col] == "k":
                isKeyAvaible = True
            if gridWorld[row][col] in directions2Ids:
                startDirection = gridWorld[row][col]
            
    for row in range(numRows):
        for col in range(numCols):
            for direction in directions2Ids:
                if gridWorld[row][col] == "W":
                    continue
                # Assign a state ID
                if not isKeyAvaible:
                    stateId += 1
                    key = 1
                    validStates = generateStatesWithKeyValue(gridWorld, row, col, direction, key, validStates, stateId, startDirection)
                else:
                    if gridWorld[row][col] != "d":
                        stateId += 1
                        key = 0
                        validStates = generateStatesWithKeyValue(gridWorld, row, col, direction, key, validStates, stateId, startDirection)
                    key = 1
                    stateId += 1
                    validStates = generateStatesWithKeyValue(gridWorld, row, col, direction, key, validStates, stateId, startDirection)                
    return validStates



def moveForward(row, col, direction, key):
    if direction == "^":
        coordinate_one = (row - 1, col, direction, key)
        coordinate_two = (row - 2, col, direction, key)
        coordinate_three = (row - 3, col, direction, key)
    elif direction == ">":
        coordinate_one = (row, col + 1, direction, key)
        coordinate_two = (row, col + 2, direction, key)
        coordinate_three = (row, col + 3, direction, key)
    elif direction == "v":
        coordinate_one = (row + 1, col, direction, key)
        coordinate_two = (row + 2, col, direction, key)
        coordinate_three = (row + 3, col, direction, key)
    elif direction == "<":
        coordinate_one = (row, col - 1, direction, key)
        coordinate_two = (row, col - 2, direction, key)
        coordinate_three = (row, col - 3, direction, key)
    return coordinate_one, coordinate_two, coordinate_three



def printTransition(currentState, nextState, action, probability, validStates):
    currentStateId = validStates[currentState]['stateID']
    nextStateId = validStates[nextState]['stateID']
    # give huge reward to goal state
    if validStates[nextState]['goal']:
        print(f'transition {currentStateId} {action} {nextStateId} 1000 {probability}')
    elif validStates[nextState]['key']:
        # give huge reward to key
        print(f'transition {currentStateId} {action} {nextStateId} 100 {probability}')
    
    else:
        # give -1 reward for staying alive
        print(f'transition {currentStateId} {action} {nextStateId} -1 {probability}')



def generateTransitions(gridWorldPath):
    numRows, numCols, gridWorld = readGridWorld(gridWorldPath)

    validStates = generateStatesFromGridWorld(numRows, numCols, gridWorld)

    numStates = len(validStates)
    print(f'numStates {numStates}')
    print('numActions 4')
    endStates = []
    for state in validStates:
        # if validStates[state]['start']:
            # print(f'start {validStates[state]["stateID"]}')
        if validStates[state]['goal']:
            endStates.append(validStates[state]['stateID'])
    
    if len(endStates) > 0:
        print(f"end {' '.join([str(x) for x in endStates])}")
    else:
        print('end -1')

    # generate transition values
            
    for state in validStates:
        row, col, direction, key = state
        stateID = validStates[state]['stateID']
        if validStates[state]['key']:
            key = 1

        if validStates[state]['goal']:
            continue
        
        state = (row, col, direction, key)
        # actions are forward, left, right, turn
        
        # ************************************************************************************************
        # ****************************************** FORWARD *********************************************
        # ************************************************************************************************
        # forward action
            
        forwardActionNeghiborsOne, forwardActionNeghiborsTwo, forwardActionNeghiborsThree = moveForward(row, col, direction, key)

        probForwardOne = 0.0
        probForwardTwo = 0.0
        probForwardThree = 0.0

        if forwardActionNeghiborsOne in validStates:
            probForwardOne = 1
            if forwardActionNeghiborsTwo in validStates:
                probForwardOne = 0.5
                probForwardTwo = 0.5
                if forwardActionNeghiborsThree in validStates:
                    probForwardOne = 0.5
                    probForwardTwo = 0.3
                    probForwardThree = 0.2
        if forwardActionNeghiborsOne in validStates:
            printTransition(state, forwardActionNeghiborsOne, action2IDs['forward'], probForwardOne, validStates)
        if forwardActionNeghiborsTwo in validStates:
            printTransition(state, forwardActionNeghiborsTwo, action2IDs['forward'], probForwardTwo, validStates)
        if forwardActionNeghiborsThree in validStates:
            printTransition(state, forwardActionNeghiborsThree, action2IDs['forward'], probForwardThree, validStates)
            

        
        # ************************************************************************************************
        # ****************************************** lEFT *********************************************
        # ************************************************************************************************
        # left action
        
        currentDirectionId = directions2Ids[direction]
        leftDirectionID = (currentDirectionId - 1) % 4
        leftDirection = ids2Directions[leftDirectionID]
        probabilityOfLeftDirection = 0.9
        leftDirectionTurnsAroundID = (currentDirectionId + 2) % 4
        leftDirectionTurnsAround = ids2Directions[leftDirectionTurnsAroundID]
        probabilityOfLeftDirectionTurnsAround = 0.1

        leftDirectionNeghibor = (row, col, leftDirection, key)
        leftDirectionTurnsAroundNeghibor = (row, col, leftDirectionTurnsAround, key)

        if leftDirectionNeghibor in validStates:
            printTransition(state, leftDirectionNeghibor, action2IDs['left'], probabilityOfLeftDirection, validStates)
        if leftDirectionTurnsAroundNeghibor in validStates:
            printTransition(state, leftDirectionTurnsAroundNeghibor, action2IDs['left'], probabilityOfLeftDirectionTurnsAround, validStates)


        # ************************************************************************************************
        # ****************************************** RIGHT *********************************************
        # ************************************************************************************************
            
        # right action
            
        rightDirectionID = (currentDirectionId + 1) % 4
        rightDirection = ids2Directions[rightDirectionID]
        probabilityOfRightDirection = 0.9
        rightDirectionTurnsAroundID = (currentDirectionId + 2) % 4
        rightDirectionTurnsAround = ids2Directions[rightDirectionTurnsAroundID]
        probabilityOfRightDirectionTurnsAround = 0.1

        rightDirectionNeghibor = (row, col, rightDirection, key)
        rightDirectionTurnsAroundNeghibor = (row, col, rightDirectionTurnsAround, key)

        if rightDirectionNeghibor in validStates:
            printTransition(state, rightDirectionNeghibor, action2IDs['right'], probabilityOfRightDirection, validStates)
        if rightDirectionTurnsAroundNeghibor in validStates:
            printTransition(state, rightDirectionTurnsAroundNeghibor, action2IDs['right'], probabilityOfRightDirectionTurnsAround, validStates)

        # ************************************************************************************************
        # ****************************************** TURN *********************************************
        # ************************************************************************************************
            
        # turn action
            
        turnDirectionID = (currentDirectionId + 2) % 4
        turnDirection = ids2Directions[turnDirectionID]
        probabilityOfTurnDirection = 0.8
        turnDirectionLeftID = (currentDirectionId - 1) % 4
        turnDirectionLeft = ids2Directions[turnDirectionLeftID]
        probabilityOfTurnDirectionLeft = 0.1
        turnDirectionRightID = (currentDirectionId + 1) % 4
        turnDirectionRight = ids2Directions[turnDirectionRightID]
        probabilityOfTurnDirectionRight = 0.1

        turnDirectionNeghibor = (row, col, turnDirection, key)
        turnDirectionLeftNeghibor = (row, col, turnDirectionLeft, key)
        turnDirectionRightNeghibor = (row, col, turnDirectionRight, key)

        if turnDirectionNeghibor in validStates:
            printTransition(state, turnDirectionNeghibor, action2IDs['turn'], probabilityOfTurnDirection, validStates)
        if turnDirectionLeftNeghibor in validStates:
            printTransition(state, turnDirectionLeftNeghibor, action2IDs['turn'], probabilityOfTurnDirectionLeft, validStates)
        if turnDirectionRightNeghibor in validStates:
            printTransition(state, turnDirectionRightNeghibor, action2IDs['turn'], probabilityOfTurnDirectionRight, validStates)

    print('mdptype episodic')
    print('discount 0.9')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gridworld', help='Path to the input grid file')
    args = parser.parse_args()
    gridfile = args.gridworld
    generateTransitions(gridfile)