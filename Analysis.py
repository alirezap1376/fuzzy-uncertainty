#t
import pyomo.environ as py
import pandas as pd
import numpy as np
import random as random
from pyomo.environ import * 


i = ['i1', 'i2', 'i3']
j = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'j10']
k = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10',
     'k11', 'k12', 'k13', 'k14', 'k15', 'k16', 'k17', 'k18', 'k19', 'k20',
     'k21', 'k22', 'k23', 'k24', 'k25', 'k26', 'k27', 'k28', 'k29', 'k30', 'k31', 'k32']


def BCCP():
    demand = { k1:np.sort( random.sample(range(10, 25), 4) ) for k1 in k}
    fixed_cost= {j1:random.randint(2000, 3500) for j1 in j }
    transport_cost_ij = {(i1, j1):np.sort( random.sample(range(1, 10), 4) ) for i1 in i for j1 in j}
    transport_cost_jk = {(j1, k1):random.randint(5, 10) for j1 in j for k1 in k }
    capacity_i =  {i1:random.randint(300, 600) for i1 in i }
    capacity_j =  {j1:random.randint(150, 500) for j1 in j } 
    Alpha = 0.7
    return demand, fixed_cost, transport_cost_ij, transport_cost_jk, capacity_i, capacity_j,Alpha 

def RPP2():
    demand = { k1:np.sort( random.sample(range(10, 25), 4) ) for k1 in k}
    fixed_cost= {j1:random.randint(2000, 3500) for j1 in j }
    transport_cost_ij = {(i1, j1):np.sort( random.sample(range(1, 10), 4) ) for i1 in i for j1 in j}
    transport_cost_jk = {(j1, k1):random.randint(5, 10) for j1 in j for k1 in k }
    capacity_i =  {i1:random.randint(300, 600) for i1 in i }
    capacity_j =  {j1:random.randint(150, 500) for j1 in j } 
    Gamma = 5
    Sigma = 8
    return demand, fixed_cost, transport_cost_ij, transport_cost_jk, capacity_i, capacity_j,Gamma, Sigma   

def RPP3():
    demand = { k1:np.sort( random.sample(range(10, 25), 4) ) for k1 in k}
    fixed_cost= {j1:random.randint(2000, 3500) for j1 in j }
    transport_cost_ij = {(i1, j1):np.sort( random.sample(range(1, 10), 4) ) for i1 in i for j1 in j}
    transport_cost_jk = {(j1, k1):random.randint(5, 10) for j1 in j for k1 in k }
    capacity_i =  {i1:random.randint(300, 600) for i1 in i }
    capacity_j =  {j1:random.randint(150, 500) for j1 in j } 
    Gamma = 5
    Sigma = 8
    return demand, fixed_cost,transport_cost_ij, transport_cost_jk,capacity_i, capacity_j,  Gamma,  Sigma  
    
def SWRPP():
    demand = { k1:np.sort( random.sample(range(10, 25), 4) ) for k1 in k}
    fixed_cost= {j1:random.randint(2000, 3500) for j1 in j }
    transport_cost_ij = {(i1, j1):np.sort( random.sample(range(1, 10), 4) ) for i1 in i for j1 in j}
    transport_cost_jk = {(j1, k1):random.randint(5, 10) for j1 in j for k1 in k }
    capacity_i =  {i1:random.randint(300, 600) for i1 in i }
    capacity_j =  {j1:random.randint(150, 500) for j1 in j } 
    Sigma = 8
    return demand, fixed_cost, transport_cost_ij, transport_cost_jk, capacity_i,  capacity_j, Sigma
    
    
    
    
    


def ModelBCCP():
    model = py.ConcreteModel()
    model.x= py.Var(i,j,domain=py.NonNegativeReals)
    model.u= py.Var(j,k,domain=py.NonNegativeReals)
    model.y= py.Var(j,domain=py.Binary)
    FirstTerm = sum(BCCP()[1][j1]*model.y[j1] for j1 in j)
    SecondTerm = sum( (sum(BCCP()[2][i1,j1])/4 )  * model.x[(i1,j1)] for i1 in i for j1 in j )
    ThirdTerm = sum( BCCP()[3][j1,k1] * model.u[(j1,k1)] for j1 in j for k1 in k)
    ExpectedValue = FirstTerm + SecondTerm + ThirdTerm
    model.obj=py.Objective(expr= ExpectedValue , sense = 1 )
    model.constraint = ConstraintList()
    for k1 in k:
        model.constraint.add( sum(model.u[(j1,k1)] for j1 in j) >= BCCP()[6]* BCCP()[0][k1][3] + (1-BCCP()[6])* BCCP()[0][k1][2] )
    for j1 in j:
        model.constraint.add( sum( model.x[(i1,j1)] for i1 in i)  ==   sum(model.u[(j1,k1)] for k1 in k) )
    for i1 in i:  
         model.constraint.add( sum( model.x[(i1,j1)] for i1 in i) <=  BCCP()[4][i1])
    for j1 in j:   
        model.constraint.add( sum( model.u[(j1,k1)] for k1 in k) <=  BCCP()[5][j1] * model.y[j1] )
    opt = py.SolverFactory('glpk' )
    opt.solve(model)
    return value(model.obj)

def ModelRPP2():
    model = py.ConcreteModel()
    model.x= py.Var(i,j,domain=py.NonNegativeReals)
    model.u= py.Var(j,k,domain=py.NonNegativeReals)
    model.y= py.Var(j,domain=py.Binary)
    model.a= py.Var(domain=py.NonNegativeReals)
    FirstTerm = sum(RPP2()[1][j1]*model.y[j1] for j1 in j)
    SecondTerm = sum( (sum(RPP2()[2][i1,j1])/4 )  * model.x[(i1,j1)] for i1 in i for j1 in j )
    ThirdTerm = sum( RPP2()[3][j1,k1] * model.u[(j1,k1)] for j1 in j for k1 in k)
    Zmax = FirstTerm + ThirdTerm + sum( RPP2()[2][i1,j1][-1] * model.x[(i1,j1)] for i1 in i for j1 in j )
    Zmin = FirstTerm + ThirdTerm + sum( RPP2()[2][i1,j1][0] * model.x[(i1,j1)] for i1 in i for j1 in j )
    ExpectedValue = FirstTerm + SecondTerm + ThirdTerm
    DevisionSirstConstraint = sum(RPP2()[0][k1][3] - (1-model.a)*RPP2()[0][k1][2] - model.a * RPP2()[0][k1][3] for k1 in k)
    model.obj=py.Objective(expr= ExpectedValue + RPP2()[6]*(Zmax-Zmin) + RPP2()[7]*(DevisionSirstConstraint) , sense = 1 )
    model.constraint = ConstraintList()
    for k1 in k:
        model.constraint.add( sum(model.u[(j1,k1)] for j1 in j) >= model.a*RPP2()[0][k1][3] + (1-model.a)*RPP2()[0][k1][2] )
    for j1 in j:
        model.constraint.add( sum( model.x[(i1,j1)] for i1 in i)  ==   sum(model.u[(j1,k1)] for k1 in k) )
    for i1 in i:  
        model.constraint.add( sum( model.x[(i1,j1)] for i1 in i) <= RPP2()[4][i1])
    for j1 in j:   
        model.constraint.add( sum( model.u[(j1,k1)] for k1 in k) <= RPP2()[5][j1] * model.y[j1] )
    model.constraint.add( model.a >= 0.5  )
    model.constraint.add( model.a <= 1  )
    opt = py.SolverFactory('glpk' )
    opt.solve(model)
    return value(model.obj)

def ModelRPP3():
    model = py.ConcreteModel()
    model.x= py.Var(i,j,domain=py.NonNegativeReals)
    model.u= py.Var(j,k,domain=py.NonNegativeReals)
    model.y= py.Var(j,domain=py.Binary)
    model.a= py.Var(domain=py.NonNegativeReals)
    FirstTerm = sum(RPP3()[1][j1]*model.y[j1] for j1 in j)
    SecondTerm = sum( (sum(RPP3()[2][i1,j1])/4 )  * model.x[(i1,j1)] for i1 in i for j1 in j )
    ThirdTerm = sum( RPP3()[3][j1,k1] * model.u[(j1,k1)] for j1 in j for k1 in k)
    Zmax = FirstTerm + ThirdTerm + sum( RPP3()[2][i1,j1][-1] * model.x[(i1,j1)] for i1 in i for j1 in j )
    ExpectedValue = FirstTerm + SecondTerm + ThirdTerm
    DevisionSirstConstraint = sum(RPP3()[0][k1][3] - (1-model.a)*RPP3()[0][k1][2] - model.a * RPP3()[0][k1][3] for k1 in k)
    model.obj=py.Objective(expr= ExpectedValue + RPP3()[6]*(Zmax) + RPP3()[7]*(DevisionSirstConstraint) , sense = 1 )
    model.constraint = ConstraintList()
    for k1 in k:
        model.constraint.add( sum(model.u[(j1,k1)] for j1 in j) >= model.a*RPP3()[0][k1][3] + (1-model.a)*RPP3()[0][k1][2] )
    for j1 in j:
        model.constraint.add( sum( model.x[(i1,j1)] for i1 in i)  ==   sum(model.u[(j1,k1)] for k1 in k) )
    for i1 in i:  
        model.constraint.add( sum( model.x[(i1,j1)] for i1 in i) <= RPP3()[4][i1])
    for j1 in j:   
        model.constraint.add( sum( model.u[(j1,k1)] for k1 in k) <= RPP3()[5][j1] * model.y[j1] )
    model.constraint.add( model.a >= 0.5  )
    model.constraint.add( model.a <= 1  )
    opt = py.SolverFactory('glpk' )
    opt.solve(model)
    return value(model.obj)    

def ModelSWRPP():
    model = py.ConcreteModel()
    model.x= py.Var(i,j,domain=py.NonNegativeReals)
    model.u= py.Var(j,k,domain=py.NonNegativeReals)
    model.y= py.Var(j,domain=py.Binary)
    model.a= py.Var(domain=py.NonNegativeReals)
    FirstTerm = sum(SWRPP()[1][j1]*model.y[j1] for j1 in j)
    SecondTerm = sum( (sum(SWRPP()[2][i1,j1])/4 )  * model.x[(i1,j1)] for i1 in i for j1 in j )
    ThirdTerm = sum( SWRPP()[3][j1,k1] * model.u[(j1,k1)] for j1 in j for k1 in k)
    Zmax = FirstTerm + ThirdTerm + sum( SWRPP()[2][i1,j1][-1] * model.x[(i1,j1)] for i1 in i for j1 in j )
    DevisionSirstConstraint =DevisionSirstConstraint = sum(SWRPP()[0][k1][3] - (1-model.a)*SWRPP()[0][k1][2] - model.a * SWRPP()[0][k1][3] for k1 in k)
    model.obj=py.Objective(expr=  Zmax + SWRPP()[6]*(DevisionSirstConstraint) , sense = 1 )
    model.constraint = ConstraintList()
    for k1 in k:
        model.constraint.add( sum(model.u[(j1,k1)] for j1 in j) >= model.a*SWRPP()[0][k1][3] + (1-model.a)*SWRPP()[0][k1][2] )
    for j1 in j:
        model.constraint.add( sum( model.x[(i1,j1)] for i1 in i)  ==   sum(model.u[(j1,k1)] for k1 in k) )
    for i1 in i:  
        model.constraint.add( sum( model.x[(i1,j1)] for i1 in i) <= SWRPP()[4][i1])
    for j1 in j:   
        model.constraint.add( sum( model.u[(j1,k1)] for k1 in k) <= SWRPP()[5][j1] * model.y[j1] )
    model.constraint.add( model.a >= 0.5  )
    model.constraint.add( model.a <= 1  )
    opt = py.SolverFactory('glpk' )
    opt.solve(model)
    return value(model.obj) 

n = 10
BPCCP_dict = dict()
RPP2_dict = dict()
RPP3_dict = dict()
SWRPP_dict = dict()

for p in range(1, n+1):
    BPCCP_dict[p] = np.nan
    RPP2_dict[p] = np.nan
    RPP3_dict[p] = np.nan
    SWRPP_dict[p] = np.nan
    
c=1
while c <= n:
    BPCCP_dict[c] = ModelBCCP()
    RPP2_dict[c] =ModelRPP2()
    RPP3_dict[c] =ModelRPP3()
    SWRPP_dict[c] = ModelSWRPP()
    c= c+1

df = pd.DataFrame(  index=list(range(1,n+1)) , columns= [ 'BPCCP' , 'RPP2', 'RPP3', 'SWRPP'   ]  )


for p in df.index :
        df.loc[p,'BPCCP'] = BPCCP_dict[p]

for p in df.index :
        df.loc[p,'RPP2'] = RPP2_dict[p]

for p in df.index :
        df.loc[p,'RPP3'] = RPP3_dict[p]

for p in df.index :
        df.loc[p,'SWRPP'] = SWRPP_dict[p]



a = df.plot()
a.set_xlabel("Iterations")
a.set_ylabel("Values")











