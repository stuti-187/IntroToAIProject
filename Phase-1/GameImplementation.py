import numpy as np
import random

depth = 9
#Initialize the game board
def createMatrix(n):
    return np.zeros(shape=(n, n))

#prints current game state
def currentGameState(g):
    print("Current Game State")
    print(g)

#color the player edge
def markVisited(n,g,s,t,player, actions):
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

    if(randomlySelectedEdge[0][1] != t):
        print("Random edge was chosen instead of input edge: "+ str(s) +" " + str(randomlySelectedEdge[0][1]))
    #t: update target accordingly
    t = randomlySelectedEdge[0][1]

    #mark these edges visited (color edge)
    g[s,t] = player
    g[t,s] = player

    #remove the colored edges from the set of valid actions
    actions.remove((s, t))
    actions.remove((t, s))

    return g

#checks if a triangle is formed by a player
def checkTriangle(n, g, s, t, player):
    if(s==None or t==None):
        return False
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


#fetch the count of edges in graph g of "player" connected to "vertex"
def degree(n, g, vertex, player):
    count = 0
    for i in range(n):
        if(g[i, vertex]==player):
            count += 1
    return (count)

#scoring method for minimax evaluation
#human player is expected to decrease the score for AI and AI will want to choose best possible move
def minmaxEvaluation(n,g):
    i = 0
    for j in range(n):
        if(degree(n, g, j, 1) == 1):
            i+=4
        #higher the degree for human better the score for AI
        if(degree(n, g, j, 1) > 1):
            i+=7
        if(degree(n, g, j, 2) == 1):
            i-=4
        #higher the degree for AI lower the score for AI
        if(degree(n, g, j, 2) > 1):
            i-=7
    return i


def maxAlphaBeta(n, g,countVisitedEdges, actions,d):
    maxv = -10000000
    s = None
    t = None
    completeGraph = (n*(n-1))/2
    if(depth==d or completeGraph < countVisitedEdges):
        return(minmaxEvaluation(n,g),0,0)
    #     return (0, 0, 0)
    elif(checkTriangle(n, g, s, t, 2)):
        return (-1, 0, 0)
    elif(checkTriangle(n, g, s, t, 1)):
        return (1, 0, 0)
    # elif(depth==d):
        # return(minmaxEvaluation(n,g),s,t)

    for i in range(n):
        for j in range(i+1,n):
            if (i,j) in actions:
                g[i, j] = 2
                g[j, i] = 2
                actions.remove((i, j))
                actions.remove((j, i))
                (currScore, minS, minT) = minAlphaBeta(n, g, countVisitedEdges+1, actions, d+1)
                if(currScore > maxv):
                    maxv = currScore
                    s = i
                    t = j
                g[i, j] = 0
                g[j, i] = 0
                actions.add((i, j))
                actions.add((j, i))

                # if (maxv >= b):
                #     return (maxv,s,t)
                #
                # if(maxv > a and d%2==0):
                #     a = maxv

    return(maxv, s, t)


def minAlphaBeta(n, g, countVisitedEdges, actions,d):
    minv = 10000000
    s = None
    t = None
    completeGraph = (n*(n-1))/2
    if(depth==d or completeGraph < countVisitedEdges):
        return(minmaxEvaluation(n,g),s,t)
    #     return (0, 0, 0)
    elif(checkTriangle(n, g, s, t, 2)):
        return (-1, 0, 0)
    elif(checkTriangle(n, g, s, t, 1)):
        return (1, 0, 0)
    # elif(depth==d):
        return(minmaxEvaluation(n,g),s,t)

    for i in range(n):
        for j in range(i+1, n):
            if (i,j) in actions:
                g[i, j] = 1
                g[j, i] = 1
                actions.remove((i, j))
                actions.remove((j, i))
                (currScore, maxS, maxT) = maxAlphaBeta(n, g,countVisitedEdges+1, actions, d+1)
                if(minv > currScore):
                    minv = currScore
                    s = i
                    t = j
                g[i, j] = 0
                g[j, i] = 0
                actions.add((i, j))
                actions.add((j, i))

                # if (minv <= a):
                #     return (minv,s,t)
                #
                # if(minv < b and d%2==1):
                #     b = minv

    return(minv, s, t)


#main
if __name__ == '__main__':
    print("Welcome to SIM Game!")
    player = 1    #1: human     2: AI
    checkWinFlag = False
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
        if player==1:
            player_move_s, player_move_t = getInput(g, player)
        else:
            (score, player_move_s, player_move_t) = maxAlphaBeta(n, g, count, actions, 0)
            print(player_move_s, " ", player_move_t)


        #if triangle is formed by a player, game ends and opposition wins
        if (checkTriangle(n, g, player_move_s, player_move_t, player)):
            if (player == 1):
                print("Player 2 wins!")
            else:
                print("Player 1 wins!")
            checkWinFlag = True
            break

        #mark the provided edge as visited, keeps track of the probability of the edge occurrence (75%-25%)
        g = markVisited(n, g, player_move_s, player_move_t, player, actions)

        print("\n")
        print("Possible Actions")
        print(actions)
        count += 1

        currentGameState(g)

        #switch to opposite player
        player = getOpposition(player)

    if (not checkWinFlag):
        print("GAME OVER! There's a tie!")
