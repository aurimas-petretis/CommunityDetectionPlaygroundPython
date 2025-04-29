# https://coin-or.github.io/pulp/main/includeme.html#quickstart

from pulp import *

x = LpVariable("x", 0, 3)
y = LpVariable("y", cat="Binary")

prob = LpProblem("myCommunityDetectionMetric", LpMinimize)
prob += x + y <= 2
prob += -4*x + y

status = prob.solve()

print(LpStatus[status])
print(x, y)
print(value(x), value(y))