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
