#RPP2

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

demand = { k1:np.sort( random.sample(range(10, 25), 4) ) for k1 in k}
fixed_cost= {j1:random.randint(2000, 3500) for j1 in j }
transport_cost_ij = {(i1, j1):np.sort( random.sample(range(1, 10), 4) ) for i1 in i for j1 in j}
transport_cost_jk = {(j1, k1):random.randint(5, 10) for j1 in j for k1 in k }
capacity_i =  {i1:random.randint(300, 600) for i1 in i }
capacity_j =  {j1:random.randint(150, 500) for j1 in j } 
Gamma = 5
Sigma = 8

model = py.ConcreteModel()

model.x= py.Var(i,j,domain=py.NonNegativeReals)
model.u= py.Var(j,k,domain=py.NonNegativeReals)
model.y= py.Var(j,domain=py.Binary)
model.a= py.Var(domain=py.NonNegativeReals)

FirstTerm = sum(fixed_cost[j1]*model.y[j1] for j1 in j)
SecondTerm = sum( (sum(transport_cost_ij[i1,j1])/4 )  * model.x[(i1,j1)] for i1 in i for j1 in j )
ThirdTerm = sum( transport_cost_jk[j1,k1] * model.u[(j1,k1)] for j1 in j for k1 in k)

Zmax = FirstTerm + ThirdTerm + sum( transport_cost_ij[i1,j1][-1] * model.x[(i1,j1)] for i1 in i for j1 in j )
Zmin = FirstTerm + ThirdTerm + sum( transport_cost_ij[i1,j1][0] * model.x[(i1,j1)] for i1 in i for j1 in j )
ExpectedValue = FirstTerm + SecondTerm + ThirdTerm
DevisionSirstConstraint = sum(demand[k1][3] - (1-model.a)*demand[k1][2] - model.a * demand[k1][3] for k1 in k)




model.obj=py.Objective(expr= ExpectedValue + Gamma*(Zmax-Zmin) + Sigma*(DevisionSirstConstraint) , sense = 1 )

model.constraint = ConstraintList()
for k1 in k:
    model.constraint.add( sum(model.u[(j1,k1)] for j1 in j) >= model.a*demand[k1][3] + (1-model.a)*demand[k1][2] )
for j1 in j:
    model.constraint.add( sum( model.x[(i1,j1)] for i1 in i)  ==   sum(model.u[(j1,k1)] for k1 in k) )
for i1 in i:  
    model.constraint.add( sum( model.x[(i1,j1)] for i1 in i) <= capacity_i[i1])
for j1 in j:   
    model.constraint.add( sum( model.u[(j1,k1)] for k1 in k) <= capacity_j[j1] * model.y[j1] )

model.constraint.add( model.a >= 0.5  )
model.constraint.add( model.a <= 1  )

opt = py.SolverFactory('glpk' )
opt.solve(model)
model.display()


            
            # Extracting decision variable values from the model
x_values = {(i1, j1): model.x[i1, j1].value for i1 in i for j1 in j}
            
            # Creating a DataFrame from the decision variable values
df_x_values = pd.DataFrame(x_values.items(), columns=['(i, j)', 'Value'])
            
            # Splitting the '(p, i)' column into separate 'p' and 'i' columns
df_x_values[['i', 'j']] = pd.DataFrame(df_x_values['(i, j)'].tolist(), index=df_x_values.index)
            
            # Dropping the '(p, i)' column
df_x_values.drop('(i, j)', axis=1, inplace=True)
            
            # Reordering columns
df_x_values = df_x_values[['i', 'j', 'Value']]



            # Extracting decision variable values from the model
u_values = {(j1, k1): model.u[j1, k1].value for j1 in j for k1 in k}
            
            # Creating a DataFrame from the decision variable values
df_u_values = pd.DataFrame(u_values.items(), columns=['(j, k)', 'Value'])
            
            # Splitting the '(p, i)' column into separate 'p' and 'i' columns
df_u_values[['j', 'k']] = pd.DataFrame(df_u_values['(j, k)'].tolist(), index=df_u_values.index)
            
            # Dropping the '(p, i)' column
df_u_values.drop('(j, k)', axis=1, inplace=True)
            
            # Reordering columns
df_u_values = df_u_values[['j', 'k', 'Value']]

model.obj.display()

model.a.display()









