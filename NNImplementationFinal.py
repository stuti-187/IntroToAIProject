import numpy as np
import torch
import TreeImplementation as treeImpl
import numbers
import random

global n
n = 5


global qnet1AdamTanH, outcomes1AdamTanH, samples1AdamTanH
class PyTorchNN(torch.nn.Module):
    
    def __init__(self, nInputs, network, nOutputs, activationFunction=3):
        super(PyTorchNN, self).__init__()
        # input layer
        networkLayers = [torch.nn.Linear(nInputs, network[0])]
        # hidden layers using the activation function (Tanh/RELU/Sigmoid)
        if len(network) > 1:
            if activationFunction==1:
                networkLayers.append(torch.nn.Tanh())
            if activationFunction==2:
                networkLayers.append(torch.nn.ReLU())
            if activationFunction==3:
                networkLayers.append(torch.nn.Sigmoid())
            
            for i in range(len(network)-1):
                networkLayers.append(torch.nn.Linear(network[i], network[i+1]))
              
                if activationFunction==1:
                    networkLayers.append(torch.nn.Tanh())
                if activationFunction==2:
                    networkLayers.append(torch.nn.ReLU())
                if activationFunction==3:
                    networkLayers.append(torch.nn.Sigmoid())
       
        # output layer
        networkLayers.append(torch.nn.Linear(network[-1], nOutputs))
        self.model = torch.nn.Sequential(*networkLayers)
        self.Xmeans = None
        self.Tmeans = None
    
    # passes data through the model
    def forward(self, X):
        return self.model(X) 
    
    # data training
    def trainPytorch(self, X, T, learningRate, nIterations):
        if self.Xmeans is None:
            self.Xmeans = X.mean(dim=0)
        if self.Tmeans is None:
            self.Tmeans = T.mean(dim=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
        lossFunc = torch.nn.MSELoss()
        trainLoss, testLoss = [], []
   
        shuffle = np.random.permutation(range(40))
        split = 10
        train, test = shuffle[:-split], shuffle[10]
        errors = []
        for iteration in range(nIterations):
            outputs = self(X)
            eTest = lossFunc(outputs[:10], T[:10])
            eTrain = lossFunc(outputs[:-10], T[:-10])
            errors.append(torch.sqrt(eTrain))
            optimizer.zero_grad()
            eTrain.backward()
            optimizer.step() 
            if iteration % 10 == 0: print("%d: %f (%f)" % (iteration, eTrain.item(), eTest.item()))
            trainLoss.append(eTrain.item() / (len(shuffle)-split))
            testLoss.append(eTest.item() / split)
        return self, errors
    
    def usePytorch(self, X):
        with torch.no_grad():
            return self(X).numpy()



def tensor(npArray):
    return torch.from_numpy(npArray.astype('double'))


def epsilonGreedy(Qnet, state, turn, actions):
    Qs = []
    actions = list(actions)
    for action in actions:
        stateActions = np.array(state+(action[0]+action[1]))
        Qs.append(Qnet.usePytorch(tensor(np.pad(stateActions, (0,37)))) if Qnet.Xmeans is not None else 0)
    Q = np.max(Qs)
    for i in range(len(Qs)):
        if((Qs[i]==Q).any()):
            move = actions[i]
    return move, Q

#  neural network training
def trainQnet(nBatches, nRepsPerBatch, network, nIterations, learningRate,activationFunction=3):
    
    Qnet = PyTorchNN(2*(n**2)+(2*n)+1, network, 1, activationFunction).double()
    repk = -1
    outcomes = np.zeros(nBatches*nRepsPerBatch)
    for batch in range(nBatches):
        
        samples = []
        
        for reps in range(nRepsPerBatch):
            repk += 1
            
            # Initialize game
            g = treeImpl.createMatrix(n)
            checkWinFlag = False
            completeGraph = (n*(n-1))/2

            actions = treeImpl.getAllValidActions(n)
            # Start game; player 1's turn initially
            player = 1
            move, _ = epsilonGreedy(Qnet, g, player,actions)
            count = 1
            # Continue to play the game
            while count <= completeGraph:
                r = 0
                player_move_s = move[0]
                player_move_t = move[1]
                g, actions = treeImpl.markVisited(n, g, move[0], move[1], player, actions)
                

                if (treeImpl.checkTriangle(n, g, player_move_s, player_move_t, player)):
                    # Determine the reinforcement
                    Qnext = 0
                    r=0
                    if (player == 1):
                        r = 2
                        outcomes[repk] = r
                    else:
                        r = 1
                        outcomes[repk] = r
                    checkWinFlag = True
                    break
                else:
                    #switch to opposite player
                    # player = treeImpl.getOpposition(player)
                    moveNext, Qnext = epsilonGreedy(Qnet, g, player, actions)
                # Collect turn results in sample array
                samples.append([*g.flatten().tolist(), move[0], move[1], r, Qnext])
                move = moveNext

        # Samples consists the training inputs and targets       
        samples = np.array(samples) 
        samples = np.pad(samples,(0,21))

        # Training inputs to the neural network
        X = tensor(samples[:-1,:]) 

        # Target values for the neural network
        T = tensor(samples[-1,:]) 

        # Training the neural network
        Qnet, _ = Qnet.trainPytorch(X, T, learningRate, nIterations) 
    
    print('DONE')
    
    return Qnet, outcomes, samples


# configuration 1
nBatches = 10
nRepsPerBatch = 16
nIterations = 100
global network
network = [3*n,n]
learningRate = 0.03

def getBestMoves(samples1AdamTanH,g,sampleCount, actions, n, count):
  allStates = samples1AdamTanH.tolist()
  res =[]
  pos = n**2
  for eachState in allStates:
    if n < 7:
      if (eachState[:pos]==g.flatten()).any():
        try:
          if not isinstance(eachState[pos+3],numbers.Integral):
            if isinstance(eachState[pos+3],np.float):
              res.append((int(eachState[pos+1]),int(eachState[pos+2])))
              res.append((int(eachState[pos+2]),int(eachState[pos+1])))
            else:
              a= eachState[pos+3].tolist()[0]
              res.append((int(eachState[pos+1]),int(eachState[pos+2])))
              res.append((int(eachState[pos+2]),int(eachState[pos+1])))
          else:
            res.append((int(eachState[pos+1]),int(eachState[pos+2])))
            res.append((int(eachState[pos+2]),int(eachState[pos+1])))
        except Exception:
          continue
      else:
        if (eachState[:pos]==g.flatten()):
          try:
            if not isinstance(eachState[pos+3],numbers.Integral):
              if isinstance(eachState[pos+3],np.float):
                res.append((int(eachState[pos+1]),int(eachState[pos+2])))
                res.append((int(eachState[pos+2]),int(eachState[pos+1])))
              else:
                a= eachState[pos+3].tolist()[0]
                res.append((int(eachState[pos+1]),int(eachState[pos+2])))
                res.append((int(eachState[pos+2]),int(eachState[pos+1])))
            else:
              res.append((int(eachState[pos+1]),int(eachState[pos+2])))
              res.append((int(eachState[pos+2]),int(eachState[pos+1])))
          except Exception:
            continue

  res = sorted(res,key=lambda x : x[1],reverse=True)

  bestMoves = set()

  for eachRes in res:
    bestMoves.add(eachRes)
  
  actionIntersection = actions.intersection(bestMoves)
  return treeImpl.chooseMove(n, g, actions, count) if len(actionIntersection)==0 else random.choices(list(actionIntersection), k=1)[0]


allLearning = []
# game starts here
def startGame():
  n_array=[4]
  for i in range(len(n_array)):
    global n,network
    
    n = n_array[i]
    # neural network configuration
    nBatches = 10
    nRepsPerBatch = 16
    nIterations = 100
    g = treeImpl.createMatrix(n)
    actions = treeImpl.getAllValidActions(n)
    count = 1
    network = [3*n,n]
    learningRate = 0.03
    
    np.set_printoptions(threshold=np.inf)
    global qnet1AdamTanH, outcomes1Adam_TanH, samples1AdamTanH
    qnet1AdamTanH, outcomes1AdamTanH, samples1AdamTanH = trainQnet(nBatches, nRepsPerBatch, network,
                                                                         nIterations, learningRate)
    
    allLearning.append(outcomes1AdamTanH)
    
allLearning = []
def gamePlay(n,g,actions,count):
    player = 1
    win2Count=0
    win1Count=0
    tieCount=0
    checkWinFlag = False
    gameSizeArray = [(0,0,0)] * 7
    completeGraph = (n*(n-1))/2
    temp = n-3
    while (count <= completeGraph):
            if player==1:
                player_move_s, player_move_t = treeImpl.chooseMove(n, g, actions, count)
            else:
                (player_move_s, player_move_t) = getBestMoves(samples1AdamTanH, g, 8,actions, n, count)
                print(player_move_s, " ", player_move_t)


            # if triangle is formed by a player, game ends and opposition wins
            if (treeImpl.checkTriangle(n, g, player_move_s, player_move_t, player)):
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

            # mark the provided edge as visited, keeps track of the probability of the edge occurrence (75%-25%)
            treeImpl.markVisited(n, g, player_move_s, player_move_t, player, actions)

            print("\n")
            print("Possible Actions")
            print(actions)
            count += 1

            treeImpl.currentGameState(g)

            # switch to opposite player
            player = treeImpl.getOpposition(player)

    if (not checkWinFlag):
        tieCount += 1
        (p1,p2,t) = gameSizeArray[temp]
        gameSizeArray[temp] = (p1,p2,t+1)
        print("GAME OVER! There's a tie!")
    print("\n")

startGame()
boardSize = 6
gamePlay(boardSize,treeImpl.createMatrix(boardSize),treeImpl.getAllValidActions(boardSize),1)