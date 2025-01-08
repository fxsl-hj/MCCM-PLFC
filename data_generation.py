import random

import numpy as np

from best_response_update_method import Response_update_method


class Data_generated:
    def __init__(self, initial_data,maximal_number_of_iterations):
        # The initial data is in dictionary form,{{},{},...,{}}
        self.initial_data = initial_data
        self.maximal_number_of_iterations = maximal_number_of_iterations

    def data_generated(self,epsilon):
        initial_data = self.initial_data
        maximal_number_of_iterations = self.maximal_number_of_iterations
        data_t_modified = {}
        data_t_suggested = {}
        cost = {}
        for index in range(len(initial_data)):
            k = 0
            data_t_modified['%d'%(index+1)] = []
            data_t_modified['%d' % (index + 1)].append(np.copy(np.array(initial_data['%d' % (index + 1)]['original_opinion'])))
            data_t_suggested['%d'%(index+1)] = []
            data_t_suggested['%d'%(index+1)].append(np.copy(np.array(initial_data['%d' % (index+1)]['suggested_opinion'])))
            cost['%d'%(index+1)] = []
            cost['%d' % (index + 1)].append(np.copy(np.array(initial_data['%d' % (index+1)]['unit_cost'])))
            while True:
                k += 1
                response_update_method = Response_update_method(initial_data['%d' % (index+1)],
                                                                maximal_number_of_iterations)
                modified_opinion, interation = response_update_method.iteration()
                data_t_modified['%d' % (index + 1)].append(np.copy(np.array(modified_opinion)))
                initial_data['%d' % (index + 1)]['original_opinion'] = np.copy(modified_opinion)
                oc_ = np.copy(np.dot(modified_opinion,np.array(initial_data['%d' % (index+1)]['weight_value']).T))
                # Determine whether consensus levels and stop conditions have been reached
                if (1-np.sum(np.absolute(oc_-np.array(modified_opinion)))/len(modified_opinion))>=epsilon or k>15:
                    break
                else:
                    z = np.random.rand(1)/2
                    initial_data['%d' % (index+1)]['suggested_opinion'] = np.copy(np.array(oc_)*z+np.array(modified_opinion)*(1-z))
                    initial_data['%d' % (index + 1)]['unit_cost'] = np.copy(np.random.rand(len(initial_data['%d' % (index+1)]['decision_makers'])))*10
                    initial_data['%d' % (index + 1)]['suggested_opinion'] = initial_data['%d' % (index+1)]['suggested_opinion'].clip(0,1)
                    data_t_suggested['%d'%(index+1)].append(np.copy(initial_data['%d' % (index+1)]['suggested_opinion']))
                    cost['%d' % (index + 1)].append(np.copy(initial_data['%d' % (index+1)]['unit_cost']))
        return data_t_modified,data_t_suggested,cost

