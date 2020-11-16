import numpy as np
import random
import matplotlib.pyplot as plt
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

def randomAI(g, actions):
    print(random.choices(list(actions), k = 1))
    return random.choices(list(actions), k = 1)[0]


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
    plt.title('Baseline AI')
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

    for i in gameSizeArray:
        print(i)


    for i in range(100):
        print("Game number: ", i + 1)
        print("Welcome to SIM Game!")
        player = 1
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
        temp = n-3
        while (count <= completeGraph): #Stopping condition: check if number of possible edges exhausted
            #player_move_s: stores the source vertex of the player
            #player_move_t: stores the target vertex of the player
            # player_move_s, player_move_t = getInput(g, player)

            if player==1:
                # player_move_s, player_move_t = getInput(g, player)
                player_move_s, player_move_t = randomAI(g, actions)
            else:
                (player_move_s, player_move_t) = randomAI(g, actions)
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
