import numpy as np
import random
import math
import matplotlib.pyplot as plt

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
def minmaxEvaluation(n,g,s, t, player):
    score = 0
    # for action in actions:
    #     s = action[0]
    #     t = action[1]
    #     score = 0
    if player==2:
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
    else:
        for i in range(n):
            if(i!=s and i!=t):
                if(g[s,i]==1 and g[i, t]==1):
                    score-=20
                if(g[s,i]==2 and g[i, t]==2):
                    score-=2
                if((g[s,i]==1 and g[i, t]==0) or (g[s,i]==0 and g[i, t]==1)):
                    score-=1
                if((g[s,i]==2 and g[i, t]==0) or (g[s,i]==0 and g[i, t]==2)):
                    score+=1
                if((g[s,i]==1 and g[i, t]==2) or (g[s,i]==2 and g[i, t]==1)):
                    score+=2
                if(g[s,i]==0 and g[i, t]==0):
                    score+=3
        return score

def minimax(n, g, maxDepth, isPlayerMaximizer, alpha, beta, countVisited, s, t, actions, player):
    if isPlayerMaximizer:
        player = 2
    else:
        player = 1
    completeGraph = n*(n-1)/2
    if(maxDepth==0 or completeGraph < countVisited or checkTriangle(n, g, s, t, player)):
        return minmaxEvaluation(n, g, s, t,player)

    if isPlayerMaximizer:
        maxValue = -1000000
        for action in actions:
            s = action[0]
            t = action[1]
            g[s,t] = 2
            g[t,s] = 2
            actions.remove((s,t))
            actions.remove((t,s))
            evaluation = minimax(n,g,maxDepth-1,False, alpha, beta, countVisited+1, s, t, actions,player)
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
            actions.remove((s,t))
            actions.remove((t,s))
            evaluation = minimax(n,g,maxDepth-1,True, alpha, beta, countVisited+1, s, t, actions,player)
            g[s,t] = 0
            g[t,s] = 0
            actions.add((s,t))
            actions.add((t,s))
            minValue = min(minValue, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return minValue


def chooseMove(n, g, actions, countVisited, player):
    moveScoreDict =  {}

    for action in actions:
        i = action[0]
        j = action[1]
        g[i,j] = 2
        g[j,i] = 2
        actions.remove((i,j))
        actions.remove((j,i))
        val = minimax(n, g, 19, True, 0,0,countVisited+1, i, j,actions, player)
        g[i,j] = 0
        g[j,i] = 0
        actions.add((i,j))
        actions.add((j,i))
        moveScoreDict[action] = val

    aiBest = max(moveScoreDict, key=moveScoreDict.get)
    return aiBest

def choosePlayerMove(n, g, actions, countVisited, player):
    moveScoreDict =  {}

    for action in actions:
        i = action[0]
        j = action[1]
        g[i,j] = 1
        g[j,i] = 1
        actions.remove((i,j))
        actions.remove((j,i))
        val = minimax(n, g, 19, False, 0,0,countVisited+1, i, j,actions, player)
        g[i,j] = 0
        g[j,i] = 0
        actions.add((i,j))
        actions.add((j,i))
        moveScoreDict[action] = val

    playerBest = max(moveScoreDict, key=moveScoreDict.get)
    return playerBest

def plottingHistogram(gameSizeArray):
    # libraries
    barWidth = 0.30
    bars1=[]
    bars2=[]
    bars3=[]
    # set height of bar
    for i in gameSizeArray:
        bars1.append(i[0])
        bars2.append(i[1])
        bars3.append(i[2])

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#1e3d59', width=barWidth, edgecolor='white', label='Player 1')
    plt.bar(r2, bars2, color='#ff6e40', width=barWidth, edgecolor='white', label='Player 2')
    plt.bar(r3, bars3, color='#ffc13b', width=barWidth, edgecolor='white', label='Tie')

    # Add xticks on the middle of the group bars
    plt.title('Minimax AI Vs Best Human Input')
    plt.ylabel('Winning Frequency', fontweight='bold')
    plt.xlabel('Board Size', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['3', '4', '5', '6', '7', '8', '9'])

    # Create legend & Show graphic
    plt.legend()
    plt.show()
#main
if __name__ == '__main__':
    i = 0
    win1Count = win2Count = tieCount = 0
    gameSizeArray = [(0,0,0)] * 7



    for i in range(100):
        print("Game number: ", i + 1)
        print("Welcome to SIM Game!")
        player = 1    #1: human     2: AI
        checkWinFlag = False
        #n: stores the input size
        # n = int(input("Enter game size: "))

        n = random.randint(3,9)
        print("Board Size: ", n)
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
        temp = n - 3
        while (count <= completeGraph): #Stopping condition: check if number of possible edges exhausted
            #player_move_s: stores the source vertex of the player
            #player_move_t: stores the target vertex of the player
            if player==1:
                # player_move_s, player_move_t = getInput(g, player)
                player_move_s, player_move_t = choosePlayerMove(n, g, actions, count,player)
            else:
                (player_move_s, player_move_t) = chooseMove(n, g, actions, count,player)
                print(player_move_s, " ", player_move_t)


            #if triangle is formed by a player, game ends and opposition wins
            if (checkTriangle(n, g, player_move_s, player_move_t, player)):
                if (player == 1):
                    win2Count += 1
                    (p1,p2,t) = gameSizeArray[temp]
                    gameSizeArray[temp] = (p1,p2+1,t)
                    print("Player 2 wins!")
                else:
                    win1Count += 1
                    (p1,p2,t) = gameSizeArray[temp]
                    gameSizeArray[temp] = (p1+1,p2,t)
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
            tieCount += 1
            (p1,p2,t) = gameSizeArray[temp]
            gameSizeArray[temp] = (p1,p2,t+1)
            print("GAME OVER! There's a tie!")
        print("\n")

    print("Player 1 win count: ", win1Count)
    print("Player 2 win count: ", win2Count)
    print("Player tie count: ", tieCount)
    j = 3
    for i in gameSizeArray:
        print("Board size: ", j, " ",  i)
        j += 1
    count = [win1Count, win2Count, tieCount]
    plottingHistogram(gameSizeArray)
