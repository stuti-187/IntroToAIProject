import numpy as np
import random

#Initialize the game board
def createMatrix(n):
    return np.zeros(shape=(n, n))

#prints current game state
def currentGameState(g):
    print("Current Game State")
    print(g)

#color the player edge
def markVisited(n,g,s,t,player):
    edgeList = []

    #creating a list of edges from source 's' to all other vertices except target 't'
    for i in range(n):
        if(g[s,i] == 0 and i != t and s != i):
            edgeList.append((s,i))

    #otherEdgeListSize: stores the number of other possible edges from s except t
    otherEdgeListSize = len(edgeList)

    edgeList.append((s,t))
    randomWeights = []

    #dividing 25% weight among other possible vertices from s
    for i in range(otherEdgeListSize):
        randomWeights.append(25/otherEdgeListSize)

    #assigning 75% weight to edge (s,t) entered by player
    randomWeights.append(75)
    randomWeightsTuple = tuple(randomWeights)

    #randomlySelectedEdge: stores the randomly selected edge based on the weights assigned
    randomlySelectedEdge = random.choices(edgeList, weights=randomWeightsTuple, k=1)

    #t: update target accordingly
    t = randomlySelectedEdge[0][1]

    #mark these edges visited (color edge)
    g[s,t] = player
    g[t,s] = player
    return g

#checks if a triangle is formed by a player
def checkTriangle(n, g, s, t, player):
    for i in range(n):
        if(g[i,s] == player and g[i,t] == player):
            return True
    return False

#return the opposition
def getOpposition(player):
    if (player == 1):
        return 2
    else:
        return 1

#return set of all valid actions in the initial state
def getAllValidActions(n):
    s = set()
    for i in range(n):
        for j in range(n):
            if i != j:
                s.add((i,j))
    return s

#fetch source and target vertices from the player
def getInput(g, player):
    player_move_s, player_move_t = input("Player " + str(player) + ": Enter source and target: ").split()
    player_move_s = int(player_move_s)
    player_move_t = int(player_move_t)
    if (player_move_s == player_move_t or g[player_move_s, player_move_t] == player):
        print("Invalid move! Try again")
        getInput(g, player)
    return (player_move_s, player_move_t)

#main
if __name__ == '__main__':
    print("Welcome to SIM Game!")
    player = 1
    #n: stores the input size
    n = int(input("Enter game size: "))
    #g: stores game grid
    g = createMatrix(n)
    #Print the initial state
    currentGameState(g)
    #actions: store all possible valid actions
    actions = getAllValidActions(n)
    print("Possible Actions")
    print(actions)

    #completeGraph: Number of possible edges in an undirected graph
    completeGraph = (n*(n-1))/2

    #count: keeps track of maximum number of edges reached
    count = 1
    print("\n")
    while (count <= completeGraph): #Stopping condition: check if number of possible edges exhausted
        #player_move_s: stores the source vertex of the player
        #player_move_t: stores the target vertex of the player
        player_move_s, player_move_t = getInput(g, player)

        #if triangle is formed by a player, game ends and opposition wins
        if (checkTriangle(n, g, player_move_s, player_move_t, player)):
            if (player == 1):
                print("Player 2 wins!")
            else:
                print("Player 1 wins!")
            break

        #mark the provided edge as visited, keeps track of the probability of the edge occurrence (75%-25%)
        g = markVisited(n, g, player_move_s, player_move_t, player)

        #remove the colored edges from the set of valid actions
        actions.remove((player_move_s, player_move_t))
        actions.remove((player_move_t, player_move_s))


        print("\n")
        print("Possible Actions")
        print(actions)
        count += 1

        currentGameState(g)

        #switch to opposite player
        player = getOpposition(player)
