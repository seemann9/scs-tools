
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
