from HGraph import *

def checkCorrectnessRandomString():
    bad = None
    for it in range(2**24):
        if it % 100 == 0:
            print(it)
            sys.stdout.flush()
        if randint(1, 2) == 1:
            g = HGraph()
            g.generate(randint(2, 3), randint(4, 11))
            if (len(g.termStrings) > 5):
                continue
        else:
            g = generateFromRandomStrings(3, 8, 3)
        lhg = len(g.getHGGreedy())
        lo = len(g.getOptimal())
        if not g.correct_falling():
            print('Fail')
            g.draw()
            print(g.termStrings)
            break
    print('End')

def checkCorrectness2RandomString():
    bad = None
    for it in range(2**24):
        if it % 100 == 0:
            print(it)
            sys.stdout.flush()
        g = generateFromRandomStrings(3, 3, 3)
        lhg = len(g.getHGGreedy(True))
        lo = len(g.getOptimal())
        if not g.correct_falling2():
            print('Fail')
            g.draw()
            print(g.termStrings)
            break
    print('End')
