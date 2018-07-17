# -*- coding: utf-8 -*-


import cvxpy as cvx
a1 = cvx.Variable()
a2 = cvx.Variable()
a3 = cvx.Variable()


constraints = [a1+a2-a3==0,a1>=0,a2>=0,a3>=0]
obj = cvx.Minimize((18*a1**2+25*a2**2+2*a3**2+42*a1*a2-12*a1*a3-14*a2*a3)/2-a1-a2-a3)

prob = cvx.Problem(obj, constraints)
prob.solve()
print(prob.status)
