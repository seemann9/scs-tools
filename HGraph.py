#!/usr/bin/python3

from pygraphviz import *
from itertools import permutations
from random import choice
from random import randint
from random import shuffle
from random import random
import sys
from os import system
import imageio
from utils import *

class HGraph:
    def __init__(self):
        self.frameCnt = 1
        self.n = 0
        self.k = 0
        self.e = []
        self.cur_edges = []

    def buildFromGraphFile(self, fname):
        f = open(fname)
        self.n, m = map(int, f.readline().strip().split())
        self.e = [[] for _ in range(self.n)]
        for i in range(m):
            u, v = map(int, f.readline().strip().split())
            self.e[u].append(v)
        self.setLabels()

    def buildFromStrings(self, strings):
        substrings = set()
        for s in strings:
            for i in range(len(s)):
                for j in range(i, len(s) + 1):
                    substrings.add(s[i:j])
        self.n = len(substrings)
        self.e = [[] for _ in range(self.n)]
        sslist = []
        for s in substrings:
            sslist.append((len(s), s))
        sslist = sorted(sslist)
        ssnum = {}
        cnt = 0
        for l, s in sslist:
            ssnum[s] = cnt
            cnt += 1
            if len(s) > 0:
                self.e[ssnum[s[:-1]]].append(ssnum[s])
                self.e[ssnum[s]].append(ssnum[s[1:]])
        self.setLabels()

    def buildFromStringsFile(self, fname):
        f = open(fname)
        strings = []
        for s in f:
            strings.append(s.strip())
        self.buildFromStrings(strings)

    def setLabels(self):
        self.ue = [[] for _ in range(self.n)]
        for v in range(self.n):
            for u in self.e[v]:
                self.ue[v].append(u)
                self.ue[u].append(v)
        self.layer = getDistances(self.n, self.ue, 0)
        self.downto = [0] * self.n
        self.downfrom = [0] * self.n
        self.term = [True] * self.n
        for v in range(self.n):
            for u in self.e[v]:
                if self.layer[u] < self.layer[v]:
                    self.term[u] = False
                    self.downto[v] = u
                else:
                    self.term[v] = False
                    self.downfrom[u] = v
        self.string = [''] * self.n
        let = 0
        for v in self.e[0]:
            self.string[v] = chr(let + ord('a'))
            let += 1
        for v in range(self.n):
            if self.layer[v] < 2:
                continue
            self.string[v] = self.string[self.downfrom[v]] + self.string[self.downto[v]][-1]
            assert(self.string[v] == self.string[self.downfrom[v]][0] + self.string[self.downto[v]])

        self.label = [''] * self.n
        self.label[0] = 'eps'
        for v in range(1, self.n):
            self.label[v] = self.string[v];

        self.tonum = {}
        for v in range(self.n):
            self.tonum[self.string[v]] = v
        self.termStrings = []
        for v in range(self.n):
            if self.term[v]:
                self.termStrings.append(self.string[v])

    def buildDrawGraph(self):
        self.drawGraph = AGraph(strict=False, directed=True)
        layers = [[] for _ in range(self.n)]
        for v in range(self.n):
            layers[self.layer[v]].append(self.label[v])
        for v in range(self.n):
            for u in self.e[v]:
                self.drawGraph.add_edge(self.label[v], self.label[u], 'direct')
        layers = reversed(layers)
        for layer in layers:
            if len(layer) > 0:
                self.drawGraph.add_subgraph(layer, rank='same')
            for i in range(len(layer) - 1):
                self.drawGraph.add_edge(layer[i], layer[i + 1], style='invis')
        for v in range(self.n):
            if self.term[v]:
                n = self.drawGraph.get_node(self.label[v])
                n.attr['shape'] = 'box'
                
    def drawEdges(self, edges, attr = {}):
        for edge, cnt in edges.items():
            if cnt == 0:
                continue
            v, u = edge
            e = self.drawGraph.get_edge(self.label[v], self.label[u], 'direct')
            for k, v in attr.items():
                e.attr[k] = v
            if cnt > 1:
                e.attr['label'] = str(cnt)

    def drawEdgesExtra(self, edges, attr = {}):
        for edge, cnt in edges.items():
            if cnt == 0:
                continue
            v, u = edge
            self.drawGraph.add_edge(self.label[v], self.label[u], color='red')
            #TODO deal with attr!!!
            #for k, v in attr.items():
            #    e.attr[k] = v

    #TODO rename
    #TODO add legends to drawings
    def draw(self, fname='graph.png'):
        self.buildDrawGraph()
        self.drawEdgesExtra(getEdgeCounterFromSequence(self.HGGreedyWalk), {'color' : 'red'})
        self.drawGraph.layout(prog='dot')

        self.drawEdges(self.fall_taken, {'color' : 'green'})

        self.drawEdges(getEdgeCounterFromList(self.cur_edges), {'color' : 'blue', 'penwidth' : '1.5'})

        self.drawGraph.draw(fname)

        #self.drawEdges(getEdgeCounterFromList(self.cur_edges), {'color' : 'black', 'penwidth' : '1.0'})
        self.cur_edges = []

    def drawGreedyAndOptimal(self, fname='graph.png'):
        self.buildDrawGraph()
        self.drawEdgesExtra(getEdgeCounterFromSequence(self.HGGreedyWalk), {'color' : 'red'})
        self.drawGraph.layout(prog='dot')
        self.drawEdges(getEdgeCounterFromSequence(self.optimalWalk), {'color' : 'green'})

        self.drawGraph.draw(fname)

    #TODO parameterize
    def drawFrame(self):
        fname = 'tmp/%03d.png' % self.frameCnt
        self.frameCnt += 1
        self.draw(fname)
        self.images.append(imageio.imread(fname))

    #Needs avconv with mp4 codecs installed on the system
    def makeVideo(self):
        system('for f in tmp/???.png; do convert $f -resize 200% $f; done')
        system('rm video.mp4')
        system('avconv -r 1 -i tmp/%03d.png -b:v 1000k video.mp4 > /dev/null 2>&1')


    def printDebug(self):
        print(self.n, 'vertices')
        print('Layer:', self.layer)
        print('Downto', self.downto)
        print('Downfrom', self.downfrom)
        print('Term', self.term)
        print('String', self.string)

    def isEulerian(self, v, startLayer, taken, todeg, fromdeg, used):
        used[v] = True
        if todeg[v] != fromdeg[v]:
            return False
        if self.layer[v] < startLayer:
            return False
        for u in self.e[v]:
            if taken[(v, u)] == 0:
                continue
            if not used[u]:
                if not self.isEulerian(u, startLayer, taken, todeg, fromdeg, used):
                    return False
        return True

    def getWalkFromTaken(self, v, taken, fromdeg):
        st = deque()
        st.append(v)
        walk = []
        while len(st) > 0:
            v = st[-1]
            if fromdeg[v] == 0:
                st.pop()
                walk.append(v)
            else:
                for u in self.e[v]:
                    if taken[(v, u)] > 0:
                        taken[(v, u)] -= 1
                        fromdeg[v] -= 1
                        st.append(u)
                        break
        return list(reversed(walk))

    def getHGGreedy(self, cycle_walk=False):
        taken = Counter()
        todeg = [0] * self.n
        fromdeg = [0] * self.n
        for v in range(self.n - 1, 0, -1):
            if self.term[v]:
                taken[(v, self.downto[v])] += 1
                fromdeg[v] += 1
                todeg[self.downto[v]] += 1
                taken[(self.downfrom[v], v)] += 1
                todeg[v] += 1
                fromdeg[self.downfrom[v]] += 1
            elif todeg[v] != fromdeg[v]:
                if todeg[v] < fromdeg[v]:
                    c = fromdeg[v] - todeg[v]
                    taken[(self.downfrom[v], v)] += c
                    fromdeg[self.downfrom[v]] += c
                    todeg[v] += c
                else:
                    c = todeg[v] - fromdeg[v]
                    taken[(v, self.downto[v])] += c
                    fromdeg[v] += c
                    todeg[self.downto[v]] += c

            elif todeg[v] + fromdeg[v] > 0:
                used = [False] * self.n
                if self.isEulerian(v, self.layer[v], taken, todeg, fromdeg, used):
                    if cycle_walk:
                        flag = False
                        for u in range(self.n):
                            if self.term[u] and not used[u]:
                                flag = True
                                break
                        if not flag:
                            break

                    taken[(v, self.downto[v])] += 1
                    fromdeg[v] += 1
                    todeg[self.downto[v]] += 1
                    taken[(self.downfrom[v], v)] += 1
                    fromdeg[self.downfrom[v]] += 1
                    todeg[v] += 1
        self.HGGreedyWalk = self.getWalkFromTaken(0 if not cycle_walk else self.n - 1, taken, fromdeg)
        res = ''
        for i in range(len(self.HGGreedyWalk) - 1):
            v = self.HGGreedyWalk[i]
            u = self.HGGreedyWalk[i + 1]
            if self.layer[u] > self.layer[v]:
                res += self.string[u][-1:]
        return res

    #TODO put in a general getWalk format
    def getCycleCover(self):
        taken = Counter()
        todeg = [0] * self.n
        fromdeg = [0] * self.n
        for v in range(self.n - 1, 0, -1):
            if self.term[v]:
                taken[(v, self.downto[v])] += 1
                fromdeg[v] += 1
                todeg[self.downto[v]] += 1
                taken[(self.downfrom[v], v)] += 1
                todeg[v] += 1
                fromdeg[self.downfrom[v]] += 1
            elif todeg[v] != fromdeg[v]:
                if todeg[v] < fromdeg[v]:
                    c = fromdeg[v] - todeg[v]
                    taken[(self.downfrom[v], v)] += c
                    fromdeg[self.downfrom[v]] += c
                    todeg[v] += c
                else:
                    c = todeg[v] - fromdeg[v]
                    taken[(v, self.downto[v])] += c
                    fromdeg[v] += c
                    todeg[self.downto[v]] += c

        return taken

    def dfsOnTaken(self, v, taken, used):
        used[v] = True
        for u in self.e[v]:
            if taken[(v, u)] > 0 and not used[u]:
                self.dfsOnTaken(u, taken, used)

    def isConnectedByTaken(self, taken):
        used = [False] *self.n
        self.dfsOnTaken(0, taken, used)
        for v in range(self.n):
            if self.term[v] and not used[v]:
                return False
        return True

    def correct_falling(self):
        taken = self.getCycleCover()
        walk = self.optimalWalk
        for i in range(len(walk) - 1):
            taken[(walk[i], walk[i + 1])] += 1
        vlist = [[self.layer[v], self.n - v - 1] for v in range(self.n)]
        vlist.sort(reverse=True)
        flist = [self.n - b - 1 for a, b in vlist]

        walk = self.HGGreedyWalk
        greedy_taken = Counter()
        for i in range(len(walk) - 1):
            e = (walk[i], walk[i + 1])
            greedy_taken[e] += 1
        prev_layer = self.layer[self.n - 1] + 1
        self.fall_taken = taken
        system('rm tmp -r')
        system('mkdir tmp')
        self.images = []
        self.drawFrame()

        for v in flist:
            if v == 0:
                break
            if self.layer[v] != prev_layer:
                cur_layer = self.layer[v]
                next_layer = cur_layer + 1
                u = v
                vertices = []
                while u < self.n and self.layer[u] == cur_layer:
                    vertices.append(u)
                    u += 1
                next_vertices = []
                while u < self.n and self.layer[u] == next_layer:
                    next_vertices.append(u)
                    u += 1
                toCnt = 0
                for u in vertices:
                    toCnt += taken[(u, self.downto[u])]
                    toCnt -= greedy_taken[(u, self.downto[u])]
                if toCnt < len(vertices):
                    return False
                for u in next_vertices:
                    if taken[(u, self.downto[u])] != greedy_taken[(u, self.downto[u])] or taken[(self.downfrom[u], u)] != greedy_taken[(self.downfrom[u], u)]:
                        return False

            prev_layer = self.layer[v]
            t = self.downto[v]
            f = self.downfrom[v]
            vt = (v, t)
            fv = (f, v)
            while taken[fv] > 0 and taken[vt] > 0:
                old_taken = Counter(taken)
                taken[fv] -= 1
                taken[vt] -= 1
                if self.layer[v] > 1:
                    u = self.downto[f]
                    assert(u == self.downfrom[t])
                    fu = (f, u)
                    ut = (u, t)
                    taken[fu] += 1
                    taken[ut] += 1
                if not self.isConnectedByTaken(taken):
                    taken[fv] += 1
                    taken[vt] += 1
                    if self.layer[v] > 1:
                        taken[fu] -= 1
                        taken[ut] -= 1
                    break
                else:
                    self.cur_edges = [vt, fv]
                    self.fall_taken = old_taken
                    self.drawFrame()

                    self.fall_taken = taken
                    if self.layer[v] > 1:
                        self.cur_edges = [fu, ut]
                    self.drawFrame()
        self.drawFrame()
        #imageio.mimsave('movie.gif', self.images)
        self.makeVideo()
        self.fall_taken = dict(taken)
        walk = self.HGGreedyWalk
        for i in range(len(walk) - 1):
            e = (walk[i], walk[i + 1])
            taken[e] -= 1
            if (taken[e] < 0):
                print(self.termStrings)
                assert(False)
        return True

    def correct_falling2(self):
        taken2 = self.getCycleCover()
        walk = self.optimalWalk
        taken = Counter()
        for i in range(len(walk) - 1):
            taken[(walk[i], walk[i + 1])] += 1
        vlist = [[self.layer[v], self.n - v - 1] for v in range(self.n)]
        vlist.sort(reverse=True)
        flist = [self.n - b - 1 for a, b in vlist]

        for v in flist:
            t = self.downto[v]
            f = self.downfrom[v]
            vt = (v, t)
            fv = (f, v)
            while taken[fv] > 0 and taken[vt] > 0:
                taken[fv] -= 1
                taken[vt] -= 1
                if self.layer[v] > 1:
                    u = self.downto[f]
                    assert(u == self.downfrom[t])
                    fu = (f, u)
                    ut = (u, t)
                    taken[fu] += 1
                    taken[ut] += 1
                if not self.isConnectedByTaken(uniteCounters(taken, taken2)):
                    taken[fv] += 1
                    taken[vt] += 1
                    if self.layer[v] > 1:
                        taken[fu] -= 1
                        taken[ut] -= 1
                    break
        taken = uniteCounters(taken, taken2)
        self.fall_taken = dict(taken)
        walk = self.HGGreedyWalk
        for i in range(len(walk) - 1):
            e = (walk[i], walk[i + 1])
            taken[e] -= 1
            if (taken[e] < 0):
                return False
        return True

    def getOptimal(self):
        terminals = [v for v in range(self.n) if self.term[v]]
        sumlen = 0
        cnt = 0
        tnum = [0] * self.n
        for v in terminals:
            tnum[v] = cnt
            cnt += 1
            sumlen += len(self.string[v])

        d = [[0] * len(terminals) for _ in terminals]
        for v in terminals:
            for u in terminals:
                d[tnum[v]][tnum[u]] = getOverlapLength(self.string[v], self.string[u])

        mn = float('inf')
        for p in permutations(terminals):
            sum = sumlen
            for i in range(len(terminals) - 1):
                sum -= d[tnum[p[i]]][tnum[p[i + 1]]]
            if sum < mn:
                mn = sum
                best = p
        res = self.string[best[0]]
        for i in range(len(best) - 1):
            t = self.string[best[i + 1]]
            res += t[d[tnum[best[i]]][tnum[best[i + 1]]]:]
        walk = [0]
        split = 0
        for j in range(len(best)):
            v = best[j]
            down = []
            u = v
            while u != split:
                down.append(u)
                u = self.downfrom[u]
            u = v
            walk.extend(reversed(down))
            for i in range(len(self.string[v]) - (d[tnum[v]][tnum[best[j + 1]]] if j < len(best) - 1 else 0)):
                u = self.downto[u]
                walk.append(u)
            split = u
        self.optimalWalk = walk
        return res

    def getGreedyString(self):
        terminals = [self.string[v] for v in range(self.n) if self.term[v]]
        while len(terminals) > 1:
            mx = -1
            for s in terminals:
                for t in terminals:
                    if s == t:
                        continue
                    cur = getOverlapLength(s, t)
                    if cur > mx:
                        mx = cur
                        best = (s, t)
            s = best[0]
            t = best[1]
            terminals.remove(s)
            terminals.remove(t)
            terminals.append(s + t[mx:])

        return terminals[0]

    #TODO describe parameters and rename
    def generate(self, k, h):
        downto = [0]
        downfrom = [0]
        self.n = k + 1
        self.e = [[]]
        prev = []
        for v in range(1, k + 1):
            self.e[0].append(v)
            self.e.append([0])
            downto.append(0)
            downfrom.append(0)
            prev.append(v)
        prevpairs = sum([[(v, u) for u in range(1, k + 1)] for v in range(1, k + 1)], [])
        for _ in range(2, h):
            sys.stdout.flush()
            cur = []
            curpairs = []
            shuffle(prevpairs)
            for (v, u) in prevpairs:
                w = self.n
                self.n += 1
                downto.append(u)
                downfrom.append(v)
                self.e.append([u])
                self.e[v].append(w)
                for x in cur:
                    if downto[x] == v:
                        curpairs.append((x, w))
                cur.append(w)
                for x in cur:
                    if downfrom[x] == u:
                        curpairs.append((w, x))
                if len(curpairs) >= k:
                    break
            prev = cur
            prevpairs = curpairs
        self.setLabels()

def generateFromRandomStrings(n, l, k):
    alph = [chr(ord('a') + i) for i in range(k)]
    ss = []
    for i in range(n):
        ss.append(''.join([choice(alph) for _ in range(l)]))
    g = HGraph()
    g.buildFromStrings(ss)
    return g



g = HGraph()
#g.read('hand1')
#TODO move to separate sample test files
#g.buildFromStrings(['aabaa', 'ababa', 'abbaa'])
#g.buildFromStrings(['abab', 'baba', 'cbba'])
#g.buildFromStrings(['aabba', 'baaab', 'baaba'])

#g.buildFromStrings(['abc', 'bca', 'cab', 'ddd'])
g.buildFromStrings(['abc', 'bca', 'cab'])
#g.buildFromStrings(['abdab', 'bacba', 'bcdb'])
#g.buildFromStrings(['aabb', 'bbaa', 'cccc'])
#g.buildFromStrings(['abab', 'abba', 'baba'])
#g.buildFromStrings(['abb', 'bbc', 'bbb'])
#g.buildFromStrings(['dbbabb', 'bbabbc', 'babbab'])
#g.buildFromStrings(['bbaa', 'aaaabaa', 'babaa'])
#g.buildFromStrings(['aabcabcabc', 'abcabcabcc', 'bcabcabcab'])
#g.generate(3, 5)
#g.buildFromStrings(['abaabab', 'baabaab', 'babaaba'])
g.printDebug()
print(g.getHGGreedy())
print(g.getOptimal())
#print(g.getGreedyString())
print(g.correct_falling())
g.draw()

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
                
                
                
#geneticSearch(4, 10, 3, 1000, 10000)
#geneticSearch(3, 20, 3, 3000, 1000)
#geneticSearch(3, 10, 3, 10, 1)
#print('Max Score:', maxScore)
#print('Best:', bestStrings)
def checkCorrectnessRandomString():
    bad = None
    for it in range(2**24):
        if it % 100 == 0:
            print(it)
            sys.stdout.flush()
        if randint(1, 2) == 1:
            g = HGraph()
    #        g.generate(3, 5)
            g.generate(randint(2, 3), randint(4, 11))
            if (len(g.termStrings) > 5):
                continue
        else:
            g = generateFromRandomStrings(3, 8, 3)
        lhg = len(g.getHGGreedy())
        lo = len(g.getOptimal())
#        lg = len(g.getGreedyString())
        if not g.correct_falling():
            print('Fail')
            g.draw()
            print(g.termStrings)
            #print('getHGGreedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)
            break
    print('End')
#checkCorrectnessRandomString()
def checkCorrectness2RandomString():
    bad = None
    for it in range(2**24):
        if it % 100 == 0:
            print(it)
            sys.stdout.flush()
        g = generateFromRandomStrings(3, 3, 3)
        lhg = len(g.getHGGreedy(True))
        lo = len(g.getOptimal())
#        lg = len(g.getGreedyString())
        if not g.correct_falling2():
            print('Fail')
            g.draw()
            print(g.termStrings)
            #print('getHGGreedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)
            break
    print('End')
#checkCorrectness2RandomString()
