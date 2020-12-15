import numpy as np
import random
import math
import NNImplementationFinal as nnif

nodesEvaluated = 0
#Initialize the game board
def createMatrix(n):
    return np.zeros(shape=(n, n))

#prints current game state
def currentGameState(g):
    print("Current Game State")
    print(g)

#color the player edge
def markVisited(n, g, s, t, player, actions):
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

    return g, actions

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

def randomAI(g, actions):
    return random.choices(list(actions), k = 1)[0]

#scoring method for minimax evaluation
#human player is expected to decrease the score for AI and AI will want to choose best possible move
def minmaxEvaluation(n,g,s, t):
    score = 0
    for i in range(n):
        if(i!=s and i!=t):
            if(g[s,i]==2 and g[i, t]==2):
                score-=20
            if(g[s,i]==1 and g[i, t]==1):
                score-=2
            if((g[s,i]==2 and g[i, t]==0) or (g[s,i]==0 and g[i, t]==2)):
                score-=1
            if((g[s,i]==1 and g[i, t]==0) or (g[s,i]==0 and g[i, t]==1)):
                score+=1
            if((g[s,i]==2 and g[i, t]==1) or (g[s,i]==1 and g[i, t]==2)):
                score+=2
            if(g[s,i]==0 and g[i, t]==0):
                score+=3
    return score

def minimax(n, g, maxDepth, isPlayerMaximizer, alpha, beta, countVisited, s, t, actions):
    global nodesEvaluated
    if isPlayerMaximizer:
        player = 2
    else:
        player = 1
    completeGraph = n*(n-1)/2
    if(maxDepth==0 or completeGraph < countVisited or checkTriangle(n, g, s, t, player)):
        return minmaxEvaluation(n, g, s, t)

    if isPlayerMaximizer:
        maxValue = -1000000
        for action in actions:
            s = action[0]
            t = action[1]
            g[s,t] = 2
            g[t,s] = 2
            nodesEvaluated += 1 
            actions.remove((s,t))
            actions.remove((t,s))
            evaluation = minimax(n,g,maxDepth-1,False, alpha, beta, countVisited+1, s, t, actions)
            g[s,t] = 0
            g[t,s] = 0
            actions.add((s,t))
            actions.add((t,s))
            maxValue = max(evaluation, maxValue)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return maxValue

    else:
        minValue = 1000000
        for action in actions:
            s = action[0]
            t = action[1]
            g[s,t] = 1
            g[t,s] = 1
            nodesEvaluated += 1 
            actions.remove((s,t))
            actions.remove((t,s))
            evaluation = minimax(n,g,maxDepth-1,True, alpha, beta, countVisited+1, s, t, actions)
            g[s,t] = 0
            g[t,s] = 0
            actions.add((s,t))
            actions.add((t,s))
            minValue = min(minValue, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return minValue


def chooseMove(n, g, actions, countVisited):
    global nodesEvaluated
    moveScoreDict =  {}

    for action in actions:
        i = action[0]
        j = action[1]
        g[i,j] = 2
        g[j,i] = 2
        nodesEvaluated += 1 
        actions.remove((i,j))
        actions.remove((j,i))
        val = minimax(n, g, 20, True, 0,0,countVisited+1, i, j,actions)
        g[i,j] = 0
        g[j,i] = 0
        actions.add((i,j))
        actions.add((j,i))
        moveScoreDict[action] = val

    best = max(moveScoreDict, key=moveScoreDict.get)
    return best
#main
if __name__ == '__main__':
    print("Welcome to SIM Game!")
    player = 1    #1: human     2: AI
    checkWinFlag = False
    opponent = int(input("Enter- \n1 to play with Human opponent \n2 to play with Baseline AI \n3 to play with Tree-based AI \n4 to play with tree+NN-based AI "))
    
    
    if opponent == 4:
      print("****************** Training the Neural Network ******************")
      nnif.startGame()
      print("*****************************************************************")
    
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

    #Game Play starts
    while (count <= completeGraph): #Stopping condition: check if number of possible edges exhausted
        #player_move_s: stores the source vertex of the player
        #player_move_t: stores the target vertex of the player


        if player==1:
            player_move_s, player_move_t = getInput(g, player)
        else:
            if opponent == 1:
                player_move_s, player_move_t = getInput(g, player)
            elif opponent == 2:
                (player_move_s, player_move_t) = randomAI(g, actions)
                print(player_move_s, " ", player_move_t)
            elif opponent == 3:
                (player_move_s, player_move_t) = chooseMove(n, g, actions, count)
                print("Number of nodes evaluated: ", nodesEvaluated)
                print(player_move_s, " ", player_move_t)
                nodesEvaluated= 0
            elif opponent == 4:
                (player_move_s, player_move_t) = nnif.getBestMoves(nnif.samples1AdamTanH, g, 8,actions, n, count)

            
        #if triangle is formed by a player, game ends and opposition wins
        if (checkTriangle(n, g, player_move_s, player_move_t, player)):
            if (player == 1):
                print("Player 2 wins!")
            else:
                print("Player 1 wins!")
            checkWinFlag = True
            break

        #mark the provided edge as visited, keeps track of the probability of the edge occurrence (75%-25%)
        g, _ = markVisited(n, g, player_move_s, player_move_t, player, actions)

        print("\n")
        print("Possible Actions")
        print(actions)
        count += 1

        currentGameState(g)

        #switch to opposite player
        player = getOpposition(player)

    if (not checkWinFlag):
        print("GAME OVER! There's a tie!")