from encoder import readGridWorld, generateStatesFromGridWorld, action2IDs, IDs2Actions, ids2Directions, directions2Ids


# IDs2Actions = {
#     0: "forward",
#     1: "left",
#     2: "right",
#     3: "turn"
# }




def getThePathForThePolicyAndValue(gridWorld, validStates, actions):
    
    startDirection = "v"
    agentHasKey = 1
    startRow, startCol = -1, -1

    numRows, numCols = len(gridWorld), len(gridWorld[0])

    for row in range(numRows):
        for col in range(numCols):
            # if gridWorld[row][col] == "s":
            #     startRow, startCol = row, col
            if gridWorld[row][col] == "k":
                agentHasKey = 0
            if gridWorld[row][col] in ["^", ">", "v", "<"]:
                startDirection = gridWorld[row][col]
                startRow, startCol = row, col
            
    startState = (startRow, startCol, startDirection, agentHasKey)
    
    startStateID = validStates[startState]['stateID']

    print(actions[startStateID] , end=" ")
    

def getGridWorldFromGridWorldFiles(gridWorldTestFile):
    fileContents = [x.strip() for x in open(gridWorldTestFile).readlines()]

    gridWorlds = []
    previousGridWorld = []
    for i in range(1, len(fileContents)):
        if fileContents[i] == "Testcase":
            if len(previousGridWorld) > 0:
                gridWorlds.append(previousGridWorld)
                previousGridWorld = []
        else:
            previousGridWorld.append([x.strip() for x in fileContents[i].split(" ")])
    gridWorlds.append(previousGridWorld)
    

    gridWorld = gridWorlds[0]
    numRows, numCols = len(gridWorld), len(gridWorld[0])

    validStates = generateStatesFromGridWorld(numRows, numCols, gridWorld)

    return gridWorlds, validStates
        



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gridworld', help='Path to the input grid file')
    parser.add_argument('--value-policy', help='Path to the value and policy file')
    parser.add_argument('--mdp', help='Path to the mdp file')
    args = parser.parse_args()

    gridWorlds, validStates = getGridWorldFromGridWorldFiles(args.gridworld)

    actions = [int(x.strip().split()[1]) for x in open(args.value_policy).readlines()]
    Values = [float(x.strip().split()[0]) for x in open(args.value_policy).readlines()]
    
    for gridWorld in gridWorlds:
        # print(*gridWorld, sep="\n")
        getThePathForThePolicyAndValue(gridWorld, validStates, actions)
    
    # getThePathForThePolicyAndValue(args.gridworld, args.value_policy)
