from searchers import *

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

searchRandomHGraphTest(1000, 5, 2)
