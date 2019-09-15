import sys                              #for maxInt
import random
from collections import defaultdict
import datetime                         #for Gantt (time formatting) #duration is in days
import plotly                           #for Gantt (full package)
import plotly.plotly as py              #for Gantt
import plotly.figure_factory as ff      #for Gantt
#plotly.tools.set_credentials_file(username='addejans', api_key='65E3LDJVN63Y0Fx0tGIQ')  #for Gantt -- DeJans pw:O********1*
plotly.tools.set_credentials_file(username='tutrinhkt94', api_key='vSrhCEpX6ADg7esoVCbc')  #for Gantt -- T.Tran pw:OaklandU
import copy                             #for saving best objects


############################## INITIALIZATION --- START

def initialization():
    random.seed(1)

#**************************************************************************#
#!!!!!!!!!!!!!!!!!!!!!!!!!!! DATA INPUT --- START !!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!! DATA ENTRY --- START !!!!!!!!!!!!!!!!!!!!!!!!!!!#
    
    global numJobs, numMachines, numNodes
    numJobs = 4
    numMachines = 3 
    numNodes = numJobs*numMachines + 2

    parameterInitialization(8,8) #input number of ants and cycles //Needed before creating node object

    #machine numbers are indexed starting at 1
    Job1MachSeq = [1,2,3]
    Job2MachSeq = [2,1,3]
    Job3MachSeq = [1,3,2]

    global Node0, Node1, Node2, Node3, Node4, Node5, Node6, Node7, Node8, Node9, Node10, Node11
    #NODE(dependents, duration, machine, nodeNum) //Duration is defined with unit as days, machine 0 is equivalent to the first machine (i.e. zero indexed)
        #the node.num property is 1 larger than the object Node# variable name
    Node0 = Node([],1,0,1)
    Node1 = Node([Node0],2,1,2)
    Node2 = Node([Node0, Node1],3,2,3)
    Node3 = Node([],4,1,4)
    Node4 = Node([Node3],5,0,5)
    Node5 = Node([Node3, Node4],6,2,6)
    Node6 = Node([],7,0,7)
    Node7 = Node([Node6],8,2,8)
    Node8 = Node([Node6, Node7],9,1,9)
    Node9 = Node([],10,0,10)
    Node10 = Node([Node9],11,1,11)
    Node11 = Node([Node9, Node10],12,2,12)
    #dummyNodes
    global Source, Sink
    Source = Node([],0,-1,0)  #The source information will not change
    sinkDependents = [Node2, Node5, Node8, Node11] #list of the last operation of each job; these will be dependents for the sink
    Sink = Node(sinkDependents,0,-1,(numNodes-1)) #The sink information will not change

    global NODESLIST
    NODESLIST = [Source,Node0,Node1,Node2,Node3,Node4,Node5,Node6,Node7,Node8,Node9,Node10,Node11,Sink] #the NODESLIST should be appended appropriately in numerical order

    Job1Nodes = [Node0,Node1,Node2]   #Nodes should be added dependent to which job they're attached
    Job2Nodes = [Node3,Node4,Node5]
    Job3Nodes = [Node6,Node7,Node8]
    Job4Nodes = [Node9,Node10,Node11]

    Job1 = Jobs([0,1,2], Job1Nodes, 1) #Jobs(jobSequence, Nodes, num) //where num refers to the job number
    Job2 = Jobs([3,4,5], Job2Nodes, 2)
    Job3 = Jobs([6,7,8], Job3Nodes, 3)
    Job4 = Jobs([9,10,11], Job4Nodes, 4)

    global JOBSLIST
    JOBSLIST = [Job1, Job2, Job3, Job4] #Append list appropriately in accordance to the jobs created.

    #final note: sequential order is necessary and required for proper functionality at this stage.
    
#!!!!!!!!!!!!!!!!!!!!!!!!!!! DATA ENTRY --- END !!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!! DATA INPUT --- END !!!!!!!!!!!!!!!!!!!!!!!!!!!#
#**************************************************************************#
    
    global ANTLIST
    ANTLIST = []

    global has_cycle
    has_cycle = False

    global smallestSoFarMakespan, bestSoFarAnt, smallestSoFarAntNum, bestSoFarANTLIST, bestSoFarNODESLIST, JOBSLIST2
    bestSoFarANTLIST = []
    bestSoFarNODESLIST = []
    JOBSLIST2 = []

    constructConjunctiveGraph()

    global solutionGraphList
    solutionGraphList = [[] for i in range(K) ]
    generateSolutionGraphs()

    global nextMachine
    nextMachine = -1

    global currentMachine
    currentMachine = -1

    global feasibleNodeLists
    feasibleNodeLists = [[] for i in range(K)]

    global T
    T = [[0.2 for i in range(numNodes)] for j in range(numNodes)] #start with a small constant?

    global H #Heuristic matrix --- will this be different for different types/"species" of ants?.. TBD ***********************************************!!!!!!!!!!!!!!!
    H = [[0.5 for i in range(numNodes)] for j in range(numNodes)] #****************** NEEDS TO BE FIXED TO WORK WITH HEURISTICS

    global machineList
    machineList = [[] for i in range(numMachines)]

    global cliquesVisited
    cliquesVisited = [[] for i in range(K)]

    generateAnts()
    generateMachineLists()

############################## INITIALIZATION --- END


############################## CLASSES --- START

class Jobs:
    def __init__(self, jobSequence, Nodes, num):
        self.jobSequence = jobSequence
        self.Nodes = Nodes
        self.num = num
        
class Node:
    def __init__(self, dependents, duration, machine, nodeNum):
        self.duration = duration
        self.dependents = [dependents for i in range(K)]
        self.machine = machine
        self.visited = False
        self.num = nodeNum
        self.startTime = 0
        self.endTime = 0
        self.scheduled = False
        self.antsVisited = [False for i in range(K)]
        self.name = 'name goes here' #fill in names via constructor
        self.discovered = False

class Ant:
    def __init__(self, num):
        self.num = num #label each ant by a number 0 to (k-1)
        self.tabu = []
        self.position = -1
        self.T = [[0 for i in range(numNodes)] for j in range(numNodes)] #pheromone matrix 
        self.pheromoneAccumulator = [[0 for i in range(numNodes)] for j in range(numNodes)] #accumulator
        self.transitionRuleMatrix = [[0 for i in range(numNodes)] for j in range(numNodes)] #for equation 1 transition probability
        self.makespan = 0
        self.species = 'none'
        self.cycleDetected = False

############################## CLASSES --- END

############################## OTHER --- START
        
def parameterInitialization(numAnts, numCycles):
    global K, C, alpha, beta, rho
    global Q
    global Q1, Q2
    alpha = 0.2 #influence of pheromone
    beta = 1 - alpha #influence of heuristic
    rho = 0.7 #evaporation constant
    K = numAnts #number of ants
    C = numCycles #number of cycles
    Q1 = float(20) #**Programming Note: this must be a float in order for TSum to calculuate as float in calculatePheromoneAccumulation()
    Q2 = float(5) 
    #EAS Procedure determination (below) of number of ants and fixed number of cycles
    #K = int(numJobs/2)
    #C = 1000
    #Q = float(5)  # //note: there is no Q1 and Q2 -- these are original

def generateAnts():
    for i in range(K): 
        ANTLIST.append(Ant(i))    

def generateMachineLists():
    for i in range(numMachines):
        for j in range(numNodes):
            if NODESLIST[j].machine == i:
                machineList[i].append(NODESLIST[j].num)
                
def generateSolutionGraphs():
    for k in range(K):
        constructConjunctiveGraph()
        solutionGraphList[k] = conjunctiveGraph
    
def constructConjunctiveGraph():
    global conjunctiveGraph
    conjunctiveGraph = [[-1 for i in range(numNodes)] for j in range(numNodes)]
    
    for job in JOBSLIST:
        for seq1 in job.jobSequence:
            for seq2 in job.jobSequence:
                if seq1 <> seq2 and seq1+1 == seq2:
                    conjunctiveGraph[seq1+1][seq2+1] = NODESLIST[seq2+1].duration
    for j in range(numJobs):
        conjunctiveGraph[Source.num][JOBSLIST[j].Nodes[0].num] = JOBSLIST[j].Nodes[0].duration
    for j in range(numJobs):
        conjunctiveGraph[JOBSLIST[j].Nodes[numMachines-1].num][Sink.num] = 0
        
def chooseClique(antNum):
    randomClique = random.randint(0,numMachines-1) #choose Clique by random - we then choose to travel on nodes based off of pheromone trails - this prevents local optimia to occur
    while randomClique in cliquesVisited[antNum]:
        randomClique = random.randint(0,numMachines-1)
    cliquesVisited[antNum].append(randomClique)
    return randomClique

def randomAssignment():
    for i in range(K):
        randNode = random.randint(1,numNodes-2)
        ANTLIST[i].tabu.append(NODESLIST[randNode].num)
        ANTLIST[i].position = randNode
        NODESLIST[randNode].antsVisited[i] = True

def defineDecidabilityRule():                                   #UNUSED 04/15/2017 -- For Heuristics **
    for ant in ANTLIST:
        speciesType = random.randint(1,2)
        if speciesType == 1:
            ant.species = 'SPT' #Shortest Processing Time (SPT)
        elif speciesType == 2:
            ant.species = 'LPT' #Longest Processing Time (LPT)
            
############################## OTHER --- END

############################## SCHEDULING --- START

def schedule(ant):
    scheduleNode(bestSoFarNODESLIST[numNodes-1],ant)
    
def scheduleNode(node,ant):
    for proceedingNode in node.dependents[ant.num]:
        if proceedingNode.scheduled == False:
            scheduleNode(proceedingNode,ant)
    positionNode(node,ant) #base case 
    node.scheduled = True

def positionNode(node,ant):
    global longestProceedingTime
    if len(node.dependents[ant.num])>0:
        node.startTime = (bestSoFarNODESLIST[node.num].dependents[ant.num][0].startTime + node.dependents[ant.num][0].duration)
        bestSoFarNODESLIST[node.num].startTime = (bestSoFarNODESLIST[node.num].dependents[ant.num][0].startTime + node.dependents[ant.num][0].duration)
        for proceedingNode in node.dependents[ant.num]:
            longestProceedingTime = (proceedingNode.startTime + proceedingNode.duration)
            if longestProceedingTime > node.startTime:
                node.startTime = longestProceedingTime
                bestSoFarNODESLIST[node.num].startTime = longestProceedingTime
    else: #node has no proceeding nodes and can be scheduled right away
        node.startTime = 0
        bestSoFarNODESLIST[node.num].startTime = 0
    node.endTime = node.startTime + node.duration
    bestSoFarNODESLIST[node.num].endTime = node.startTime + node.duration
############################## SCHEDULING --- END


############################## END OF CYCLE --- START

def calculatePheromoneAccumulation(ant,b):
    if b == 0:
        for i in range(numNodes):
            for j in range(numNodes):
                if i != j and solutionGraphList[ant.num][i][j] > 0:
                    ant.pheromoneAccumulator[i][j] = Q1/ant.makespan # calculate w.r.t. equation 3 or 4 *** //Python Note: in python 2.7 one of the two integers
                                                                    # must be a float, we declared Q as a float() type in the parameterInitialization() method
    elif b == 1:
        print 'look here see'
        for i in range(numNodes):
            print solutionGraphList[ant.num][i]
            for j in range(numNodes):
                if i != j and solutionGraphList[ant.num][i][j] > 0:
                    ant.pheromoneAccumulator[i][j] = Q2/ant.makespan 
                                                                    
def updatePheromone(bestMakespan, bestAntNum):
    TSum = 0
    TOld = T
    for i in range(numNodes):
        for j in range(numNodes):
            for ant in ANTLIST:
                    TSum += ant.pheromoneAccumulator[i][j]
            T[i][j] = TSum + rho*TOld[i][j] #update T[i][j] pheromone matrix based on equation 2 ***          ***c<1 accounted for in construction() [main] method***
            #pheromoneAccumulator[i][j] = 0   <---- Necessary?
            TSum = 0
    for i in range(numNodes):   #update for best ant
        for j in range(numNodes):
            if solutionGraphList[bestAntNum][i][j] > 0:
                T[i][j] += float(float(solutionGraphList[bestAntNum][i][j])/float(bestMakespan))

def resetAnts():
    nextMachine = -1
    for k in range(K):
        for i in range(numMachines):
            cliquesVisited[k].pop()
    constructConjunctiveGraph()
    generateSolutionGraphs()
    #**LEARNING REMARK: #cliquesVisited = [[] for i in range(K)] *** NOTE: In Python this does not reset the variable/ double array list
    for ant in ANTLIST:
        ant.tabu = []
        ant.position = -1
        ant.T = [[0 for i in range(numNodes)] for j in range(numNodes)] #pheromone matrix 
        ant.pheromoneAccumulator = [[0 for i in range(numNodes)] for j in range(numNodes)] #accumulator
        ant.makespan = 0
        ant.cycleDetected = False
    currentMachine = -1
        
def resetNodes():
    for k in range(K):
        for node in NODESLIST:
            node.visited = False
            node.antsVisited[k] = False
    for k in range(K):
        Node0.dependents[k]=[] #**LEARNING REMARK: #We can't overwrite objects
        Node1.dependents[k]=[Node0]
        Node2.dependents[k]=[Node0, Node1]
        Node3.dependents[k]=[]
        Node4.dependents[k]=[Node3]
        Node5.dependents[k]=[Node3, Node4]
        Node6.dependents[k]=[]
        Node7.dependents[k]=[Node6]
        Node8.dependents[k]=[Node6, Node7]
        Node9.dependents[k]=[]
        Node10.dependents[k]=[Node9]
        Node11.dependents[k]=[Node9, Node10]
        #dummyNodes: (Source & Sink)
        Source.dependents[k] = []
        Sink.dependents[k] = [Node2, Node5, Node8, Node11]
############################## END OF CYCLE --- END

############################## EXPLORATION --- START
        
def nextOperation(ant, machNum, cycle):
    findFeasibleNodes(ant, machNum)
    calculateTransitionProbability(ant)
    makeDecision(ant)
    
def findFeasibleNodes(ant,currentMachine):
    global feasibleNodeLists
    feasibleNodeLists = [[] for i in range(K)]
    for node in NODESLIST:
        if node.antsVisited[ant.num] == False: #04/04/17: Removed not()
            if node.num in machineList[currentMachine]:
                feasibleNodeLists[ant.num].append(node)

def calculateTransitionProbability(ant):
    for node in feasibleNodeLists[ant.num]:
        if node.num not in ant.tabu:
            ant.transitionRuleMatrix[ant.position][node.num] = (((T[ant.position][node.num])**(alpha)) * ((H[ant.position][node.num])**(beta)))/sum((((T[ant.position][l.num])**(alpha)) * ((H[ant.position][l.num])**(beta))) for l in feasibleNodeLists[ant.num])

def makeDecision(ant):
    probabilityList = []                        #assign ranges between [0,1] and pick a rand num in interval.
    for node in feasibleNodeLists[ant.num]:
        probabilityList.append([ant.transitionRuleMatrix[ant.position][node.num]*100,node.num])
    for i in range(len(probabilityList)-1):
        probabilityList[i+1][0] += probabilityList[i][0]
    randomSelection = random.randint(0,100)
    selectedNode = -1
    for i in range(len(probabilityList)-1):
        if (probabilityList[i][0] <= randomSelection) and (randomSelection <= probabilityList[i+1][0]):
            selectedNode = probabilityList[i+1][1]
            break
        elif randomSelection <= probabilityList[i][0]: #should this be a strict less than, "<" ?
            selectedNode = probabilityList[i][1]
            break
    if selectedNode == -1:
        selectedNode = probabilityList[0][1]
    ant.position = selectedNode
         
############################## EXPLORATON --- END
            
############################## CONSTRUCTION PHASE --- START

def constructionPhase(): #The Probabilistic Construction Phrase of solutions begins by K ants
    for c in range(C):
        for i in range(numNodes):#loop for printing purposes
            print T[i]
            #print ("{0:.15f}".format(round(float(a),2)))
        print '\n \n \n'
        defineDecidabilityRule()
        for node in NODESLIST: #reset nodes visited to false since this is a new cycle
            for ant in ANTLIST:
                node.antsVisited[ant.num] = False
        for ant in ANTLIST:
            skip_counter = 0
            for i in range(numMachines):
                currentMachine = chooseClique(ant.num) #at random
                if c<1:
                    shuffledNodes = machineList[currentMachine] #We could shuffle this list or we could just keep as is
                    for x in machineList[currentMachine]:
                        if skip_counter%numJobs != 0:
                            oldAntPosition = ant.position
                        ant.tabu.append(x)
                        ant.position = x
                        if skip_counter%numJobs == 0:
                            NODESLIST[ant.position].antsVisited[ant.num] = True
                        if skip_counter%numJobs != 0:
                            NODESLIST[ant.position].antsVisited[ant.num] = True
                            NODESLIST[ant.position].dependents[ant.num].append(NODESLIST[oldAntPosition])
                            solutionGraphList[ant.num][oldAntPosition][ant.position] = NODESLIST[ant.position].duration
                        skip_counter += 1
                else:
                    for j in range(len(machineList[currentMachine])):
                        if skip_counter%numJobs != 0:
                            moveFrom = ant.position
                        nextOperation(ant, currentMachine, c)
                        moveTo = ant.position
                        ant.tabu.append(moveTo)
                        NODESLIST[moveTo].visited = True
                        NODESLIST[moveTo].antsVisited[ant.num] = True
                        if skip_counter%numJobs != 0:
                            NODESLIST[moveTo].dependents[ant.num].append(NODESLIST[moveFrom])
                            solutionGraphList[ant.num][moveFrom][moveTo] = NODESLIST[moveTo].duration # set equal to the duration of the moving to node (puts weight on edge)
                        skip_counter += 1
        for ant in ANTLIST:
            if c == 0:
                for j in range(numNodes):
                    print solutionGraphList[ant.num][j]
                print 'The c=0 makespan is: ', ant.makespan
                print '\n \n \n'
                
            undiscoverNodes()
            global has_cycle
            has_cycle = False
            cycleDetector(ant)
            ant.cycleDetected = has_cycle
            if ant.cycleDetected == False:
                ant.makespan = getMakespan(ant) #longest path in solution graph --- this is equivalent to the makespan
                calculatePheromoneAccumulation(ant,0)
            elif ant.cycleDetected == True:
                ant.makespan = sys.float_info.max
                #print(' need to fill this out, so this way pheromones and stuff are filled properly for cycle ants, i.e. they contribute nothing.') #**************************************!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        smallestMakespan = sys.float_info.max
        smallestMakespan,smallestMakespanAntNum = getSmallestMakespan()
        calculatePheromoneAccumulation(ANTLIST[smallestMakespanAntNum],1) #reinforce the smallestMakespan ant
        print 'The best makespan of cycle ' + str(c) + ' is: ', smallestMakespan
        print ANTLIST[smallestMakespanAntNum].makespan
        print ANTLIST[smallestMakespanAntNum].cycleDetected
        for j in range(numNodes):
            print solutionGraphList[smallestMakespanAntNum][j]
        if c>0:
            if smallestMakespan < smallestSoFarMakespan:
                bestSoFarAnt = copy.deepcopy(ANTLIST[smallestMakespanAntNum])
                for i in range(numNodes):
                    bestSoFarNODESLIST[i] = copy.deepcopy(NODESLIST[i])
                for i in range(K):
                    bestSoFarANTLIST[i] = copy.deepcopy(ANTLIST[i])
                for i in range(numJobs):
                    JOBSLIST2[i] = copy.deepcopy(JOBSLIST[i])
                smallestSoFarMakespan = smallestMakespan
                smallestSoFarAntNum = smallestMakespanAntNum
        elif c == 0:
            bestSoFarAnt = copy.deepcopy(ANTLIST[smallestMakespanAntNum])
            for i in range(numNodes):
                bestSoFarNODESLIST.append(copy.deepcopy(NODESLIST[i]))
            for i in range(K):
                bestSoFarANTLIST.append(copy.deepcopy(ANTLIST[i]))
            for i in range(numJobs):
                JOBSLIST2.append(copy.deepcopy(JOBSLIST[i]))
            smallestSoFarMakespan = smallestMakespan
            smallestSoFarAntNum = smallestMakespanAntNum
        updatePheromone(smallestMakespan, smallestMakespanAntNum)
        resetNodes()
        resetAnts()
        print 'bestSoFarAnt.makespan: =', bestSoFarAnt.makespan
        for i in range(numNodes):
            print i,':'
            for j in range(len(bestSoFarNODESLIST[i].dependents[bestSoFarAnt.num])):
                print bestSoFarNODESLIST[i].dependents[bestSoFarAnt.num][j].num
            print '\n'
    schedule(bestSoFarAnt)
    print 'bestSoFarAnt.makespan: =', bestSoFarAnt.makespan
    for i in range(numNodes):
        print i,':'
        for j in range(len(bestSoFarNODESLIST[i].dependents[bestSoFarAnt.num])):
            print bestSoFarNODESLIST[i].dependents[bestSoFarAnt.num][j].num
        print '\n'
    for i in range(numNodes):
        print i,':'
        print 'num',bestSoFarNODESLIST[i].num
        print 'start',bestSoFarNODESLIST[i].startTime
        print 'end',bestSoFarNODESLIST[i].endTime
        print '\n'
    makeGanttChart(bestSoFarAnt)

############################## CONSTRUCTION PHASE --- END

############################## USED FOR DETECTING CYCLES --- START
def cycleDetector(ant):
    global has_cycle
    for node in NODESLIST:
        undiscoverNodes() #sets all nodes to undiscovered
        pcount = 0
        S = []  #let S be a stack
        S.append(node)
        while len(S) > 0:
            v = S.pop()
            if v.discovered == False:
                if v != node:
                    v.discovered = True
                if v == node and pcount >=1:
                    has_cycle = True
                    return
                for j in range(numNodes):
                    if solutionGraphList[ant.num][v.num][j] >= 0:
                        S.append(NODESLIST[j])
            pcount += 1

def undiscoverNodes():
    for node in NODESLIST:
        node.discovered = False

############################## USED FOR DETECTING CYCLES --- END


############################## MAKESPAN --- START

def getSmallestMakespan():
    smallestMakespan = sys.float_info.max #1.7976931348623157e+308
    smallestMakespanAntNum = -1
    for ant in ANTLIST:
        if ant.makespan < smallestMakespan:
            smallestMakespan = ant.makespan
            smallestMakespanAntNum = ant.num
    return smallestMakespan, smallestMakespanAntNum

def getMakespan(ant):    
    G = defaultdict(list)
    edges = []
    for i in range(numNodes):
        for j in range(numNodes):
            if solutionGraphList[ant.num][i][j] != -1:
                edges.append([NODESLIST[i], NODESLIST[j]])  
    for (s,t) in edges:
        G[s].append(t)
    all_paths = DFS(G,Source)
    max_len = 0
    max_paths = []
    max_makespan = 0
    path_duration = 0
    mkspnIndex_i = -1
    for i in range(len(all_paths)):
        path_duration = 0
        for j in range(len(all_paths[i])):
            path_duration += all_paths[i][j].duration
        if path_duration > max_makespan:
            max_makespan = path_duration
            mkspnIndex_i = i
    return max_makespan

def DFS(G,v,seen=None,path=None): #v is the starting node
    if seen is None: seen = []
    if path is None: path = [v]
    seen.append(v)
    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths

############################## USED FOR GETTING MAKESPAN --- END

############################## GANTT --- START
def makeGanttChart(bestSoFarAnt):
    #quit()  #to not use more than 30 API calls per hour or 50 per day
    df = []
    for job in JOBSLIST: #I believe this could stay as JOBSLIST since we pass in only the node.num into the bestSoFarNODESLIST
        for node in job.Nodes:
            print node.num
            print job.num
            s = str(datetime.datetime.strptime('2017-04-18 00:00:00', "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=bestSoFarNODESLIST[node.num].startTime))
            d = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=bestSoFarNODESLIST[node.num].duration)
            df.append(dict(Task=str(node.machine), Start=str(s), Finish=str(d), Resource=str(job.num)))
            print s
            print d

    colors = {'1': 'rgb(220, 0, 0)',
              '2': (1, 0.9, 0.16),
              '3': 'rgb(0, 255, 100)',
              '4': 'rgb(100, 200, 50)'}
    quit()
    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
    py.iplot(fig, filename='4J:3M; param(' + str(K) + ',' + str(C) + ')-makespan: ' + str(bestSoFarAnt.makespan) , world_readable=True)
    print 'Schedule completed.'

############################## GANTT --- END

############################## Main() Method Below:

initialization()
constructionPhase()
print 'Schedule completed.'

'''
Idea for interface:
When an ant moves to a node, add dependency to node it's moving from if it's not already a dependent.
Give node a position propery for matrix. e.g., node.matrixPosition = [i][j] #look into paper about using modulus
'''
