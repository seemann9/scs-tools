from collections import deque
from collections import Counter
from pygraphviz import *
from itertools import permutations
from random import choice
from random import randint
from random import shuffle
from random import random
import sys
from os import system
import imageio

#vertex number 0 - eps
MAXN = 100
MAXK = 5

def bfs(n, e, st):
    used = [False] * n
    d = [float('inf')] * n
    d[st] = 0
    q = deque()
    q.append(st)
    used[st] = True
    while len(q) > 0:
        v = q.popleft()
        for u in e[v]:
            if not used[u]:
                used[u] = True
                q.append(u)
                d[u] = d[v] + 1
    return d

def overlap(s1, s2):
    for l in range(min(len(s1), len(s2)) - 1, 0, -1):
        if s1[-l:] == s2[:l]:
            return l
    return 0

def uniteCounters(a, b):
    res = Counter()
    for k, v in a.items():
        res[k] += v
    for k, v in b.items():
        res[k] += v
    return res

class HG:
    def __init__(self):
        self.frameCnt = 1
        self.n = 0
        self.k = 0
        self.e = []
        self.cur_edges = []

    def read(self, fname):
        f = open(fname)
        self.n, m = map(int, f.readline().strip().split())
        self.e = [[] for _ in range(self.n)]
        for i in range(m):
            u, v = map(int, f.readline().strip().split())
            self.e[u].append(v)
        g.mark()

    def fromStrings(self, strings):
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
        self.mark()

    def visualize(self, fname='graph.png'):
        G = AGraph(strict=False, directed=True)
        layers = [[] for _ in range(self.n)]
        for v in range(self.n):
            #self.string[v] = v
            layers[self.layer[v]].append(self.string[v] if v != 0 else 'eps')
        for v in range(self.n):
            for u in self.e[v]:
                G.add_edge(self.string[v] if v != 0 else 'eps', self.string[u] if u != 0 else 'eps', 'direct')
        layers = reversed(layers)
        for layer in layers:
            if len(layer) > 0:
                G.add_subgraph(layer, rank='same')
            for i in range(len(layer) - 1):
                G.add_edge(layer[i], layer[i + 1], style='invis')
        for v in range(self.n):
            if self.term[v]:
                n = G.get_node(self.string[v] if v != 0 else 'eps')
                n.attr['shape'] = 'box'
        for i in range(0, len(self.hg_greedy_walk) - 1):
            v = self.hg_greedy_walk[i]
            u = self.hg_greedy_walk[i + 1]
            G.add_edge(self.string[v] if v != 0 else 'eps', self.string[u] if u != 0 else 'eps',color='red')

        G.layout(prog='dot')
#        for i in range(0, len(self.optimal_walk) - 1):
#            v = self.optimal_walk[i]
#            u = self.optimal_walk[i + 1]
#            G.get_edge(self.string[v] if v != 0 else 'eps', self.string[u] if u != 0 else 'eps').attr['color'] = 'green'
        for v in range(self.n):
            for u in self.e[v]:
                if self.fall_taken[(v, u)] > 0:
                    e = G.get_edge(self.string[v] if v != 0 else 'eps', self.string[u] if u != 0 else 'eps', 'direct')
                    e.attr['color'] = 'green'
                    if self.fall_taken[(v, u)] > 1:
                        e.attr['label'] = str(self.fall_taken[(v, u)])
        for v, u in self.cur_edges:
            e = G.get_edge(self.string[v] if v != 0 else 'eps', self.string[u] if u != 0 else 'eps', 'direct')
            e.attr['color'] = 'blue'
            e.attr['penwidth'] = '1.5'


#        print([v.attr['pos'] for v in G.nodes()])
        G.draw(fname)
        for v, u in self.cur_edges:
            e = G.get_edge(self.string[v] if v != 0 else 'eps', self.string[u] if u != 0 else 'eps', 'direct')
            e.attr['color'] = 'black'
            e.attr['penwidth'] = '1.0'
        self.cur_edges = []


    def mark(self):
        self.ue = [[] for _ in range(self.n)]
        for v in range(self.n):
            for u in self.e[v]:
                self.ue[v].append(u)
                self.ue[u].append(v)
        self.layer = bfs(self.n, self.ue, 0)
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

        self.tonum = {}
        for v in range(self.n):
            self.tonum[self.string[v]] = v
        self.termStrings = []
        for v in range(self.n):
            if self.term[v]:
                self.termStrings.append(self.string[v])

    def print(self):
        print(self.n, 'vertices')
        print('Layer:', self.layer)
        print('Downto', self.downto)
        print('Downfrom', self.downfrom)
        print('Term', self.term)
        print('String', self.string)

    def eulerian(self, v, start_layer, taken, todeg, fromdeg, used):
        used[v] = True
        if todeg[v] != fromdeg[v]:
            return False
        if self.layer[v] < start_layer:
            return False
        for u in self.e[v]:
            if taken[(v, u)] == 0:
                continue
            if not used[u]:
                if not self.eulerian(u, start_layer, taken, todeg, fromdeg, used):
                    return False
        return True

    def get_walk(self, v, taken, fromdeg):
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

    def hg_greedy(self, cycle_walk=False):
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
                if self.eulerian(v, self.layer[v], taken, todeg, fromdeg, used):
                    if cycle_walk:
                        flag = False
                        for u in range(self.n):
                            if self.term[u] and not used[u]:
                                flag = True
                                break
                        if not flag:
#                            print('Break')
#                            print(v)
#                            print(used)
                            break

                    taken[(v, self.downto[v])] += 1
                    fromdeg[v] += 1
                    todeg[self.downto[v]] += 1
                    taken[(self.downfrom[v], v)] += 1
                    fromdeg[self.downfrom[v]] += 1
                    todeg[v] += 1
#        print(taken)
#        print(fromdeg)
#        print(todeg)
        self.hg_greedy_walk = self.get_walk(0 if not cycle_walk else self.n - 1, taken, fromdeg)
#        print(self.hg_greedy_walk)
        res = ''
        for i in range(len(self.hg_greedy_walk) - 1):
            v = self.hg_greedy_walk[i]
            u = self.hg_greedy_walk[i + 1]
            if self.layer[u] > self.layer[v]:
                res += self.string[u][-1:]
        return res
    def cycle_cover(self):
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

    def dfs(self, v, taken, used):
        used[v] = True
        for u in self.e[v]:
            if taken[(v, u)] > 0 and not used[u]:
                self.dfs(u, taken, used)

    def connected(self, taken):
        used = [False] *self.n
        self.dfs(0, taken, used)
        for v in range(self.n):
            if self.term[v] and not used[v]:
                return False
        return True

    def drawFrame(self):
            fname = 'tmp/%03d.png' % self.frameCnt
            self.frameCnt += 1
            self.visualize(fname)
            self.images.append(imageio.imread(fname))

    def makeVideo(self):
        system('for f in tmp/???.png; do convert $f -resize 200% $f; done')
        system('rm video.mp4')
        system('avconv -r 1 -i tmp/%03d.png -b:v 1000k video.mp4 > /dev/null 2>&1')

    def correct_falling(self):
        taken = self.cycle_cover()
        walk = self.optimal_walk
        for i in range(len(walk) - 1):
            taken[(walk[i], walk[i + 1])] += 1
        vlist = [[self.layer[v], self.n - v - 1] for v in range(self.n)]
        vlist.sort(reverse=True)
        flist = [self.n - b - 1 for a, b in vlist]

        walk = self.hg_greedy_walk
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
                if not self.connected(taken):
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
        walk = self.hg_greedy_walk
        for i in range(len(walk) - 1):
            e = (walk[i], walk[i + 1])
            taken[e] -= 1
            if (taken[e] < 0):
                print(self.termStrings)
                assert(False)
        return True

    def correct_falling2(self):
        taken2 = self.cycle_cover()
        walk = self.optimal_walk
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
                if not self.connected(uniteCounters(taken, taken2)):
                    taken[fv] += 1
                    taken[vt] += 1
                    if self.layer[v] > 1:
                        taken[fu] -= 1
                        taken[ut] -= 1
                    break
        taken = uniteCounters(taken, taken2)
        self.fall_taken = dict(taken)
        walk = self.hg_greedy_walk
        for i in range(len(walk) - 1):
            e = (walk[i], walk[i + 1])
            taken[e] -= 1
            if (taken[e] < 0):
                return False
        return True

    def optimal(self):
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
                d[tnum[v]][tnum[u]] = overlap(self.string[v], self.string[u])

#        d = [[overlap(self.string[v], self.string[u]) if self.term[v] and self.term[u] else 0 for u in range(self.n)] for v in range(self.n)]
        mn = float('inf')
        for p in permutations(terminals):
            sum = sumlen
            for i in range(len(terminals) - 1):
                sum -= d[tnum[p[i]]][tnum[p[i + 1]]]
            if sum < mn:
                mn = sum
                best = p
        #TODO
        #best = (27, 25, 28)
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
        self.optimal_walk = walk
        return res

    def greedy(self):
        terminals = [self.string[v] for v in range(self.n) if self.term[v]]
        while len(terminals) > 1:
            mx = -1
            for s in terminals:
                for t in terminals:
                    if s == t:
                        continue
                    cur = overlap(s, t)
                    if cur > mx:
                        mx = cur
                        best = (s, t)
            s = best[0]
            t = best[1]
            terminals.remove(s)
            terminals.remove(t)
            terminals.append(s + t[mx:])

        return terminals[0]

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
        self.mark()

def generateFromRandomStrings(n, l, k):
    alph = [chr(ord('a') + i) for i in range(k)]
    ss = []
    for i in range(n):
        ss.append(''.join([choice(alph) for _ in range(l)]))
    g = HG()
    g.fromStrings(ss)
    return g



g = HG()
#g.read('hand1')
#g.fromStrings(['aabaa', 'ababa', 'abbaa'])
#g.fromStrings(['abab', 'baba', 'cbba'])
#g.fromStrings(['aabba', 'baaab', 'baaba'])

#g.fromStrings(['abc', 'bca', 'cab', 'ddd'])
g.fromStrings(['abc', 'bca', 'cab'])
#g.fromStrings(['abdab', 'bacba', 'bcdb'])
#g.fromStrings(['aabb', 'bbaa', 'cccc'])
#g.fromStrings(['abab', 'abba', 'baba'])
#g.fromStrings(['abb', 'bbc', 'bbb'])
#g.fromStrings(['dbbabb', 'bbabbc', 'babbab'])
#g.fromStrings(['bbaa', 'aaaabaa', 'babaa'])
#g.fromStrings(['aabcabcabc', 'abcabcabcc', 'bcabcabcab'])
#g.generate(3, 5)
#g.fromStrings(['abaabab', 'baabaab', 'babaaba'])
g.print()
print(g.hg_greedy())
print(g.optimal())
#print(g.greedy())
print(g.correct_falling())
g.visualize()
def individualStrings(lenString, individual):
    return [individual[lenString * i:lenString*(i + 1)] for i in range(len(individual) // lenString)]
maxScore = 0
evaluated = {}
def evaluateFitness(s, ss):
    global maxScore, bestStrings
    if ss in evaluated:
        return evaluated[ss]
    g = HG()
    g.fromStrings(s)
    lhg = len(g.hg_greedy())
#    lg = len(g.greedy())
    lo = len(g.optimal())
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
            g = HG()
    #        g.generate(3, 5)
            g.generate(randint(2, 3), randint(4, 11))
            if (len(g.termStrings) > 5):
                continue
        else:
            g = generateFromRandomStrings(3, 8, 3)
        lhg = len(g.hg_greedy())
        lo = len(g.optimal())
#        lg = len(g.greedy())
        if not g.correct_falling():
            print('Fail')
            g.visualize()
            print(g.termStrings)
            #print('hg_greedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)
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
        lhg = len(g.hg_greedy(True))
        lo = len(g.optimal())
#        lg = len(g.greedy())
        if not g.correct_falling2():
            print('Fail')
            g.visualize()
            print(g.termStrings)
            #print('hg_greedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)
            break
    print('End')
#checkCorrectness2RandomString()



def searchRandomStringTest():
    mx = 0
    for it in range(2**24):
        if it % 100 == 0:
            print(it)
            sys.stdout.flush()
        g = generateFromRandomStrings(5, 5, 2)
        lhg = len(g.hg_greedy())
        lo = len(g.optimal())
        lg = len(g.greedy())
    #    print("Iteration", it, 'hg_greedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)
        if (lhg / lo > mx):
            mx = lhg / lo
            best = g.termStrings

    g = HG()
    g.fromStrings(best)
    lhg = len(g.hg_greedy())
    lo = len(g.optimal())
    lg = len(g.greedy())
    g.visualize()
    print('Best:', best)
    print('hg_greedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)

def searchRandomHGTest():
    mx = 0
    for it in range(1000000):
        if it % 1000 == 0:
            print(it)
            sys.stdout.flush()
        g = HG()
#        g.generate(3, 5)
        g.generate(2, randint(4, 11))
        if (len(g.termStrings) > 5):
            continue
        lhg = len(g.hg_greedy())
        lo = len(g.optimal())
        lg = len(g.greedy())
    #    print("Iteration", it, 'hg_greedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)
        if (lhg / lo > mx):
            mx = lhg / lo
            best = g.termStrings

    g = HG()
    g.fromStrings(best)
    lhg = len(g.hg_greedy())
    lo = len(g.optimal())
    lg = len(g.greedy())
    g.visualize()
    print('Best:', best)
    print('hg_greedy:',lhg, 'optimal:', lo, 'greedy:', lg, 'score:', lhg / lo)

#searchRandomStringTest()
#searchRandomHGTest()
