# https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html

import pulp as pl

solver_list = pl.listSolvers(onlyAvailable=True)
print(solver_list)

model = pl.LpProblem("Example", pl.LpMinimize)
# solver = pl.CPLEX_CMD() # need to setup with env variables if wanted
solver = pl.PULP_CBC_CMD()
_var = pl.LpVariable('a')
_var2 = pl.LpVariable('a2')
model += _var + _var2 == 1
result = model.solve(solver)