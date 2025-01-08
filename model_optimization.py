import numpy as np
import matplotlib.pyplot as plt
from DE_algorithm import DE_algorithom
import pandas as pd
import json

decision_makers_total = []
for i in range(15):
    decision_makers_total.append(i + 1)
original_opinion = np.array(
    [0.6626, 0.0954, 0.7309, 0.0547, 0.6875, 0.4851, 0.0674, 0.7827, 0.7531, 0.2770, 0.0220, 0.9281, 0.8693, 0.3486,
     0.8098])
weight_value = np.array(
    [0.0182, 0.0909, 0.0364, 0.0727, 0.0909, 0.0909, 0.0727, 0.0545, 0.0182, 0.0909, 0.0727, 0.0909, 0.0727, 0.0545,
     0.0729])
# Import the learned parameters
with open('parameters.txt','r',encoding='UTF-8') as f:
    data = f.readline().strip()
    para = json.loads(data)
get_parameters = para
# Basic information about the decision scenario
initial_information = {'decision_makers': decision_makers_total,
                       'original_opinion': original_opinion,
                       'weight_value': weight_value,
                       'defined_utility_function_parameters': get_parameters}
result = []
o_s_and_c_all = []
modified_opinion_all = []
iteration_number = 5001
optimization_model_solve = DE_algorithom(iteration_number, initial_information, 20, 0.85)
# compensation strategy optimization
result_storage, modified_opinion, o_s_and_c, particle_result = optimization_model_solve.training(100, 0.4, 0.8, 0.2)
result.append(result_storage)
o_s_and_c_all.append(o_s_and_c)
modified_opinion_all.append(modified_opinion)
np.set_printoptions(suppress=True)
columns_label1 = []
columns_label2 = []
columns_label3 = []
for i in range(iteration_number):
    columns_label1.append('%d' % (i + 1))
for i in range(30):
    if i < 15:
        columns_label2.append('c%d' % (i + 1))
    else:
        columns_label2.append('o*%d' % (i - 14))
for i in range(15):
    columns_label3.append('0_%d' % (i + 1))
result_transform = pd.DataFrame(result, columns=columns_label1)
result_transform.to_csv("data1-1.csv")
o_s_and_c = pd.DataFrame(o_s_and_c_all, columns=columns_label2)
o_s_and_c.to_csv("data1-2.csv")
modified_opinion = pd.DataFrame(modified_opinion_all, columns=columns_label3)
modified_opinion.to_csv("data1-3.csv")
particle_transform = pd.DataFrame(particle_result, columns=columns_label2)
particle_transform.to_csv("data1-4.csv")
