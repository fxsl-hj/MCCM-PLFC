from particle_swarm_optimization import Particle_swarm_optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

Absolute_error = []
initial_data = []
initial_data_generated = {}
defined_utility_function_parameters = np.random.rand(15,3).round(4)
defined_utility_function_parameters[:,1] = -defined_utility_function_parameters[:,1]
defined_utility_function_parameters[:,2] = -defined_utility_function_parameters[:,2]
np.set_printoptions(suppress=True)
decision_makers_single_total = 0
# Generate simulation data for 300 decision scenarios
for index in range(300):
    defined_utility_function_parameters_z = np.copy(defined_utility_function_parameters)
    decision_makers_single = np.random.randint(10, 16, 1)
    decision_makers_single = decision_makers_single[0]
    decision_makers_single_total += decision_makers_single
    zqz = []
    for z in range(decision_makers_single):
        zk = len(zqz)
        while True:
            zq = np.random.randint(1, 16, 1)
            zq = zq[0]
            if z == 0:
                zqz.append(zq)
                break
            for l in range(len(zqz)):
                if zq == zqz[l]:
                    break
                elif l == len(zqz) - 1:
                    zqz.append(zq)
            if len(zqz) == zk + 1:
                break
    decision_makers_total =[]
    for z in range(len(zqz)):
        q = min(zqz)
        zqz.remove(q)
        decision_makers_total.append(q)
    weight_value1 = np.random.randint(1,6,len(decision_makers_total))
    weight_value = weight_value1 / np.sum(weight_value1)
    unit_cost = np.random.rand(decision_makers_single)*10
    original_opinion = np.random.rand(decision_makers_single).round(4)
    z = np.random.rand(1)/2
    suggested_opinion = (original_opinion + z* (np.dot(original_opinion, weight_value.T) - original_opinion)).round(
        4)
    defined_utility_function_parameters_single = []
    for z in range(decision_makers_single):
        if z == 0:
            defined_utility_function_parameters_single = [defined_utility_function_parameters[decision_makers_total[z]-1]]
        else:
            defined_utility_function_parameters_single.append(
                defined_utility_function_parameters[decision_makers_total[z]-1])
    initial_data_generated_ = {'decision_makers': decision_makers_total,
                               'original_opinion': original_opinion,
                               'weight_value': weight_value,
                               'suggested_opinion': suggested_opinion,
                               'unit_cost': unit_cost,
                               'defined_utility_function_parameters': defined_utility_function_parameters_single}
    initial_data_generated['%d' % (index + 1)] = initial_data_generated_
    initial_data_index_ = np.array([decision_makers_total, original_opinion, suggested_opinion, unit_cost, weight_value]).T
    initial_data_index = np.hstack((initial_data_index_, defined_utility_function_parameters_single))
    initial_data.append(initial_data_index)
initial_data_z = initial_data[0]
for z in range(299):
    initial_data_z = np.hstack((initial_data_z.T, initial_data[z + 1].T)).T
initial_data_transform = pd.DataFrame(initial_data_z, columns=['decision_makers',
                                                               'original_opinion',
                                                               'suggested_opinion',
                                                               'unit_cost',
                                                               'weight_value',
                                                               'parameter1',
                                                               'parameter2',
                                                               'parameter3'])
# save simulation data
initial_data_transform.to_csv("initial_data.csv")
C1 = 0.85
C2 = 0.002
boundary_value = 10
swarm_optimization_operator = Particle_swarm_optimization(initial_data_generated,
                                                          boundary_value,
                                                          C1,
                                                          C2,
                                                          defined_utility_function_parameters)
c1 = 0.5
c2 = 2
c3 = 2
particle_number = 100
iteration_number = 151
total1, total2, ff_best,S = swarm_optimization_operator.get_and_iteration_particle(c1, c2, c3, particle_number, iteration_number)
np.set_printoptions(suppress=True)
z = total1 - defined_utility_function_parameters
Absolute_error.append(np.sum(np.absolute(total1 - defined_utility_function_parameters)))
for i in range(300):
    S['s%d'%(i+1)]['Dk'] = S['s%d'%(i+1)]['Dk'].tolist()
    S['s%d'%(i+1)]['Ok'] = S['s%d'%(i+1)]['Ok'].tolist()
    S['s%d'%(i+1)]['Osk'] = S['s%d'%(i+1)]['Osk'].tolist()
    S['s%d'%(i+1)]['Ck'] = S['s%d'%(i+1)]['Ck'].tolist()
if isinstance(S, str):
    S = eval(S)
with open('S.txt', 'w', encoding='UTF-8') as f:
    S_ = json.dumps(S, ensure_ascii=False)
    f.write(S_)
if isinstance(total2, str):
    total2 = eval(total2)
with open('R.txt', 'w', encoding='UTF-8') as f:
    total2_ = json.dumps(total2,ensure_ascii=False)
    f.write(total2_)
Absolute_error1 = np.sum(Absolute_error) / 5
Absolute_error.append(Absolute_error1)
columns_label= []
for i in range(6):
    columns_label.append(i+1)
Absolute_error = [Absolute_error]
Absolute_error_transform = pd.DataFrame(Absolute_error, columns=columns_label)
Absolute_error_transform.to_csv("Absolute_error.csv")
