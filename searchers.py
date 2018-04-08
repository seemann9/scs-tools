from HGraph import *

def searchRandomStringTest(iterations, size, length, alphabet = 2):
    mx = 0
    for it in range(iterations):
        if it % 100 == 0:
            print(it)
            sys.stdout.flush()
        g = generateFromRandomStrings(size, length, alphabet)
        lhg = len(g.getHGGreedy())
        lo = len(g.getOptimal())
        lg = len(g.getGreedyString())
        if (lhg / lo > mx):
            mx = lhg / lo
            best = g.termStrings

    g = HGraph()
    g.buildFromStrings(best)
    lhg = len(g.getHGGreedy())
    lo = len(g.getOptimal())
    lg = len(g.getGreedyString())
    g.drawGreedyAndOptimal()
    print('Best:', best)
    print('getHGGreedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)

def searchRandomHGraphTest(iterations, height, alphabet = 2):
    mx = 0
    for it in range(iterations):
        if it % 1000 == 0:
            print(it)
            sys.stdout.flush()
        g = HGraph()
        g.generate(alphabet, height)
        if (len(g.termStrings) > 5):
            continue
        lhg = len(g.getHGGreedy())
        lo = len(g.getOptimal())
        lg = len(g.getGreedyString())
        if (lhg / lo > mx):
            mx = lhg / lo
            best = g.termStrings

    g = HGraph()
    g.buildFromStrings(best)
    lhg = len(g.getHGGreedy())
    lo = len(g.getOptimal())
    lg = len(g.getGreedyString())
    g.drawGreedyAndOptimal()
    print('Best:', best)
    print('getHGGreedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)

def individualStrings(lenString, individual):
    return [individual[lenString * i:lenString*(i + 1)] for i in range(len(individual) // lenString)]
maxScore = 0
evaluated = {}
def evaluateFitness(s, ss):
    global maxScore, bestStrings
    if ss in evaluated:
        return evaluated[ss]
    g = HGraph()
    g.buildFromStrings(s)
    lhg = len(g.getHGGreedy())
#    lg = len(g.getGreedyString())
    lo = len(g.getOptimal())
    score = lhg / lo
    if score > maxScore:
        maxScore = score
        bestStrings = s[:]
#    score = 25**(lhg / lo - 1)
#    score = lhg / lo - 0.95
    evaluated[ss] = score
    return score
    
CXPB = 0.5
MUTPB = 0.2

def kcross(k, s, t):
    splitPoints = sorted([randint(0, len(s)) for _ in range(k)] + [0] + [len(s)])
    ss = ''
    tt = ''
    a = [s, t]
    n = 0
    for i in range(len(splitPoints) - 1):
        ss += a[n][splitPoints[i]:splitPoints[i + 1]]
        tt += a[1 - n][splitPoints[i]:splitPoints[i + 1]]
        n = 1 - n
    return ss, tt
def mutate(s, alph):
    p = randint(0, len(s) - 1)
    c = alph[:]
    c.remove(s[p])
    return s[:p] + choice(c) + s[p + 1:]


def geneticSearch(numStrings, lenString, alphSize, popSize, numGenerations):
    lenIndividual = numStrings * lenString
    alph = [chr(i + ord('a')) for i in range(alphSize)]
    population = []
    while len(population) < 0:
        ss = ''.join([choice(alph) for i in range(lenIndividual)])
        s = individualStrings(lenString, ss)
        score = evaluateFitness(s, ss) 
        if (score > 1.05):
            population.append(ss)
            print(len(population))
            sys.stdout.flush()
    while len(population) < popSize:
        ss = ''.join([choice(alph) for i in range(lenIndividual)])
        population.append(ss)
            
#    population = [''.join([choice(alph) for i in range(lenIndividual)]) for _ in range(popSize)]
    for it in range(numGenerations):
        weighted = []
        for i in population:
            weighted.append([evaluateFitness(individualStrings(lenString, i), i), i])
        sumWeights = 0
        maxWeight = 0
        for w, i in weighted:
            sumWeights += w
            maxWeight = max(w, maxWeight)
        print('Generation', it, 'Average score:', sumWeights / popSize, 'Max score:', maxWeight)
        sys.stdout.flush()
        sumWeights = 0
        for p in weighted:
            p[0] = 3000 ** (p[0] - 1)
            sumWeights += p[0]
        for p in weighted:
            p[0] /= sumWeights
        weighted.sort()
        weighted.reverse()
        prefixWeights = [0]
        cursum = 0
        for w, i in weighted:
            cursum += w
            prefixWeights.append(cursum)
        offspring = []
        for i in range(popSize):
            p = random()
            l = 0
            r = popSize
            while l < r - 1:
                m = (l + r) //2
                if prefixWeights[m] > p:
                    r = m
                else:
                    l = m
            offspring.append(weighted[l][1][:])
        temp = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random() < CXPB:
                child1, child2 = kcross(numStrings, child1, child2)
            temp.append(child1)
            temp.append(child2)
        offspring = temp
        temp = []
        for child in offspring:
            if random() < MUTPB:
                child = mutate(child, alph)
            temp.append(child)
        offspring = temp
        population = offspring[:]
