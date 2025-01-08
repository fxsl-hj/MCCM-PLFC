import numpy as np

from utility_function import Utility_function


class Data_processing:
    def __init__(self,S,d):
        self.S = S
        self.d = d

    def data_processing(self):
        S = self.S
        d = self.d
        data_numbers = len(S)
        real_data_total = []
        contrast_data_1_total = []
        contrast_data_2_total = []
        l_i = {'real_data':{},'contrast_data_1':{},'contrast_data_2':{}}
        li = {}
        for k in range(15):
            l_i['real_data']['%d' % (k+1)]= np.array([[1], [1], [1],[1]])
            l_i['contrast_data_1']['%d' % (k+1)] = np.array([[1], [1], [1]])
            l_i['contrast_data_2']['%d' % (k+1)] = np.array([[1], [1], [1]])
            li['%d'%(k+1)] = 0
        # Generate utility data corresponding to preference rank set
        for s_index in range(data_numbers):
            decision_makers_number = len(S['s%d'%(s_index+1)]['Dk'])
            w = np.ones(decision_makers_number,dtype=int)/decision_makers_number
            utility_function = Utility_function(S['s%d'%(s_index+1)],w,decision_makers_number,d['%d'%(s_index+1)]['defined_utility_function_parameters'])
            real_data, contrast_data_1, contrast_data_2 = utility_function.total_function()
            real_data_total.append(real_data)
            contrast_data_1_total.append(contrast_data_1)
            contrast_data_2_total.append(contrast_data_2)
            for D_index in range(decision_makers_number):
                z = 0
                for k in range(len(real_data[0][D_index][0])):
                    if not (real_data[0][D_index][0][k] ==-100 and real_data[0][D_index][1][k]==-100 and real_data[0][D_index][2][k]==-100):
                        z += 1
                        l_i['real_data']['%d'%S['s%d'%(s_index+1)]['Dk'][D_index]] = \
                            np.copy(np.hstack((l_i['real_data']['%d'%S['s%d'%(s_index+1)]['Dk'][D_index]],[[real_data[0][D_index][0][k]],[real_data[0][D_index][1][k]],[real_data[0][D_index][2][k]],[real_data[0][D_index][3][k]]])))
                        l_i['contrast_data_1']['%d'%S['s%d'%(s_index+1)]['Dk'][D_index]] = \
                            np.copy(np.hstack((l_i['contrast_data_1']['%d'%S['s%d'%(s_index+1)]['Dk'][D_index]],[[contrast_data_1[0][D_index][0][k]],[contrast_data_1[0][D_index][1][k]],[contrast_data_1[0][D_index][2][k]]])))
                        l_i['contrast_data_2']['%d'%S['s%d'%(s_index+1)]['Dk'][D_index]] = \
                            np.copy(np.hstack((l_i['contrast_data_2']['%d'%S['s%d'%(s_index+1)]['Dk'][D_index]],[[contrast_data_2[0][D_index][0][k]],[contrast_data_2[0][D_index][1][k]],[contrast_data_2[0][D_index][2][k]]])))

                li['%d'%S['s%d'%(s_index+1)]['Dk'][D_index]] += z
        return real_data_total,contrast_data_1_total,contrast_data_2_total,l_i,li