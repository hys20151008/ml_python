# -*- coding: utf-8 -*-

import cvxpy as cvx

w1 = cvx.Variable()
w2 = cvx.Variable()
b  = cvx.Variable()

#约束条件
constraints = [3*w1+3*w2+b>=1,4*w1+3*w2+b>=1,-w1-w2-b>=1]
#目标函数
obj = cvx.Minimize((w1**2+w2**2)/2)


prob = cvx.Problem(obj, constraints)
prob.solve()


print("w1=",w1.value)
print("w2=",w2.value)
print("b=",b.value)

