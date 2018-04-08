#!/usr/bin/python3
from collections import deque
from collections import Counter

#just general BFS
def getDistances(verticesCount, edgeLists, startNumber):
    used = [False] * verticesCount
    d = [float('inf')] * verticesCount
    d[startNumber] = 0
    q = deque()
    q.append(startNumber)
    used[startNumber] = True
    while len(q) > 0:
        v = q.popleft()
        for u in edgeLists[v]:
            if not used[u]:
                used[u] = True
                q.append(u)
                d[u] = d[v] + 1
    return d

#naive quadratic overlap
def getOverlapLength(s1, s2):
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

def getEdgeCounterFromSequence(seq):
    edges = Counter()
    for i in range(0, len(seq) - 1):
        v = seq[i]
        u = seq[i + 1]
        edges[(v, u)] += 1
    return edges

def getEdgeCounterFromList(lst):
    edges = Counter()
    for v, u in lst:
        edges[(v, u)] += 1
    return edges
