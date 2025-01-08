import math

import numpy as np

from utility_function import Utility_function


class Response_update_method:
    def __init__(self,
                 initial_information,
                 maximal_number_of_iterations):
        # initial_informationis dictionary form, {decision makers, original_opinion, suggested_opinion, unit cost, weight, parameter}
        self.decision_makers = initial_information['decision_makers']
        self.original_opinion = initial_information['original_opinion']
        self.suggested_opinion = initial_information['suggested_opinion']
        self.unit_cost = initial_information['unit_cost']
        self.weight_value = initial_information['weight_value']
        self.defined_utility_function_parameters = initial_information['defined_utility_function_parameters']
        self.maximal_number_of_iterations = maximal_number_of_iterations

    # Nash equilibrium solution
    def iteration(self):
        decision_makers = self.decision_makers
        decision_makers_numbers = len(decision_makers)
        original_opinion = self.original_opinion
        suggested_opinion = self.suggested_opinion
        unit_cost = self.unit_cost
        weight_value = self.weight_value
        defined_utility_function_parameters = self.defined_utility_function_parameters
        maximal_number_of_iterations = self.maximal_number_of_iterations
        iteration_k = 0
        modified_opinion = np.copy(original_opinion)
        modified_opinion_k = {'1':np.copy(original_opinion)}
        for iteration_index in range(maximal_number_of_iterations):
            iteration_k += 1
            modified_opinion_k['%d'%(iteration_k+1)]=[]
            wio_j_ = 0
            for index_1 in range(decision_makers_numbers):
                wio_j_ += weight_value[index_1] * modified_opinion[index_1]
            for decision_makers_index in range(decision_makers_numbers):
                z1_total = {}
                z2_total = {}
                wio_j = np.copy(wio_j_-weight_value[decision_makers_index] * modified_opinion[decision_makers_index])
                for index_2 in range(decision_makers_numbers):
                    if index_2 != decision_makers_index and original_opinion[decision_makers_index]<=suggested_opinion[decision_makers_index]:
                        z1 = original_opinion[decision_makers_index]+np.absolute(np.array(original_opinion[index_2]-modified_opinion[index_2])*((unit_cost[index_2]/unit_cost[decision_makers_index])**(1/2)))
                        z1_total['%d' % (index_2 + 1)] = z1
                        if original_opinion[decision_makers_index]<z1 <suggested_opinion[decision_makers_index]:
                            z2_total['%d' % (index_2 + 1)] =z1
                    elif index_2 != decision_makers_index and original_opinion[decision_makers_index]>suggested_opinion[decision_makers_index]:
                        z1 = original_opinion[decision_makers_index]-np.absolute(np.array(original_opinion[index_2]-modified_opinion[index_2])*((unit_cost[index_2]/unit_cost[decision_makers_index])**(1/2)))
                        z1_total['%d' % (index_2+1)] = z1
                        if original_opinion[decision_makers_index]>z1>suggested_opinion[decision_makers_index]:
                            z2_total['%d' % (index_2 + 1)] = z1
                z_sorted1 = sorted(z1_total.items(),key=lambda x :x[1])
                z_sorted2 = sorted(z2_total.items(),key=lambda x :x[1])
                alpha = defined_utility_function_parameters[decision_makers_index][0]
                beta = defined_utility_function_parameters[decision_makers_index][1]
                lambda_ = defined_utility_function_parameters[decision_makers_index][2]
                z = np.copy(modified_opinion[decision_makers_index])
                # When the original opinion is less than or equal to the suggested opinion
                if original_opinion[decision_makers_index] <= suggested_opinion[decision_makers_index]:
                    max_segmented = 0
                    max_total = []
                    for i1 in range(len(z_sorted2)+1):
                        o1 = 0
                        o2 = 0
                        if i1 == 0:
                            o1 = original_opinion[decision_makers_index]
                        else:
                            o1 = z_sorted2[i1 - 1][1]
                        if i1 == len(z_sorted2):
                            o2 = suggested_opinion[decision_makers_index]
                        else:
                            o2 = z_sorted2[i1][1]
                        decision_makers_1 = []
                        decision_makers_2 = []
                        for i2 in range(len(z_sorted1)-i1):
                            decision_makers_1.append(z_sorted1[i2+i1][0])
                        for i2 in range(i1):
                            decision_makers_2.append(z_sorted1[i2][0])
                        a = alpha*unit_cost[decision_makers_index]*original_opinion[decision_makers_index]+\
                                              (beta*(len(decision_makers_2)-2*len(decision_makers_1))*unit_cost[decision_makers_index]*original_opinion[decision_makers_index])/(decision_makers_numbers-1)+\
                                              lambda_*weight_value[decision_makers_index]*(original_opinion[decision_makers_index]-wio_j)
                        b = alpha*unit_cost[decision_makers_index]+\
                                               (beta*(len(decision_makers_2)-2*len(decision_makers_1))*unit_cost[decision_makers_index])/(decision_makers_numbers-1)+ \
                                               lambda_*math.pow(weight_value[decision_makers_index],2)
                        f_k_3 = -1000000
                        extreme_value_point = 0
                        if b != 0:
                            extreme_value_point = a/b
                            if b > 0 or\
                               extreme_value_point < min(o1,o2) or\
                               extreme_value_point > max(o1,o2):
                                f_k_3 = -1000000
                            else:
                                modified_opinion[decision_makers_index] = extreme_value_point
                                f_k_3 = Response_update_method.utility_function(unit_cost,
                                                                                original_opinion,
                                                                                modified_opinion,
                                                                                defined_utility_function_parameters[decision_makers_index],
                                                                                decision_makers_index,
                                                                                weight_value)
                        modified_opinion[decision_makers_index] = np.copy(o1)
                        f_k_1 = Response_update_method.utility_function(unit_cost,
                                                                        original_opinion,
                                                                        modified_opinion,
                                                                        defined_utility_function_parameters[decision_makers_index],
                                                                        decision_makers_index,
                                                                        weight_value)
                        modified_opinion[decision_makers_index] = np.copy(o2)
                        f_k_2 = Response_update_method.utility_function(unit_cost,
                                                                        original_opinion,
                                                                        modified_opinion,
                                                                        defined_utility_function_parameters[decision_makers_index],
                                                                        decision_makers_index,
                                                                        weight_value)
                        f_k = [f_k_1, f_k_2, f_k_3]
                        o_k = [o1,o2,extreme_value_point]
                        ML_k = np.argmax(f_k)+1
                        max_total.append(f_k[ML_k-1])
                        if i1==0:
                            max_segmented = o_k[ML_k-1]
                        if f_k[ML_k-1]>= np.max(max_total):
                            max_segmented = o_k[ML_k-1]
                        modified_opinion[decision_makers_index] = np.copy(z)
                    modified_opinion_k['%d'%(iteration_k+1)].append(max_segmented)
                # When the original opinion is more than the suggested opinion
                if original_opinion[decision_makers_index] > suggested_opinion[decision_makers_index]:
                    max_segmented = 0
                    max_total=[]
                    for i1 in range(len(z_sorted2)+1):
                        o1 = 0
                        o2 = 0
                        if i1 == 0:
                            o2 = original_opinion[decision_makers_index]
                        else:
                            o2 = z_sorted2[len(z_sorted2)-i1][1]
                        if i1 == len(z_sorted2):
                            o1 = suggested_opinion[decision_makers_index]
                        else:
                            o1 = z_sorted2[len(z_sorted2)-i1-1][1]
                        decision_makers_1 = []
                        decision_makers_2 = []
                        for i2 in range(len(z_sorted1)-i1):
                            decision_makers_1.append(z_sorted1[len(z_sorted1)-i2-i1-1][0])
                        for i2 in range(i1):
                            decision_makers_2.append(z_sorted1[len(z_sorted1)-i2-1][0])
                        a = alpha*unit_cost[decision_makers_index]*original_opinion[decision_makers_index]+\
                                              (beta*(len(decision_makers_2)-2*len(decision_makers_1))*unit_cost[decision_makers_index]*original_opinion[decision_makers_index])/(decision_makers_numbers-1)+\
                                              lambda_*weight_value[decision_makers_index]*(original_opinion[decision_makers_index]-wio_j)
                        b = alpha*unit_cost[decision_makers_index]+\
                                               (beta*(len(decision_makers_2)-2*len(decision_makers_1))*unit_cost[decision_makers_index])/(decision_makers_numbers-1)+ \
                                               lambda_*math.pow(weight_value[decision_makers_index],2)
                        f_k_3 = -1000000
                        extreme_value_point = 0
                        if b != 0:
                            extreme_value_point = a/b
                            if b>0 or\
                               extreme_value_point < min(o1,o2) or\
                               extreme_value_point > max(o1,o2):
                                f_k_3 = -1000000
                            else:
                                modified_opinion[decision_makers_index] = extreme_value_point
                                f_k_3 = Response_update_method.utility_function(unit_cost,
                                                                                original_opinion,
                                                                                modified_opinion,
                                                                                defined_utility_function_parameters[decision_makers_index],
                                                                                decision_makers_index,
                                                                                weight_value)
                        modified_opinion[decision_makers_index] = np.copy(o1)
                        f_k_1 = Response_update_method.utility_function(unit_cost,
                                                                        original_opinion,
                                                                        modified_opinion,
                                                                        defined_utility_function_parameters[decision_makers_index],
                                                                        decision_makers_index,
                                                                        weight_value)
                        modified_opinion[decision_makers_index] = np.copy(o2)
                        f_k_2 = Response_update_method.utility_function(unit_cost,
                                                                        original_opinion,
                                                                        modified_opinion,
                                                                        defined_utility_function_parameters[decision_makers_index],
                                                                        decision_makers_index,
                                                                        weight_value)
                        f_k = [f_k_1, f_k_2, f_k_3]
                        o_k = [o1,o2,extreme_value_point]
                        ML_k = np.argmax(f_k)+1
                        max_total.append(f_k[ML_k-1])
                        if i1==0:
                            max_segmented = np.copy(o_k[ML_k-1])
                        if f_k[ML_k-1]>= np.max(max_total):
                            max_segmented = np.copy(o_k[ML_k-1])
                        modified_opinion[decision_makers_index] = np.copy(z)
                    modified_opinion_k['%d'%(iteration_k+1)].append(max_segmented)
            o_k_1_k = np.copy(np.array(modified_opinion_k['%d'%(iteration_k+1)])-modified_opinion)
            if np.sum(np.absolute(o_k_1_k))>0 and iteration_k<maximal_number_of_iterations:
                modified_opinion = np.copy(modified_opinion_k['%d' % (iteration_k + 1)])
            else:
                modified_opinion = np.copy(modified_opinion_k['%d' % iteration_k])
                break
        return modified_opinion,iteration_k

    @staticmethod
    def utility_function(c,o, o_,alpha_beta_gamma_lambda,decision_make,weight_value):
        number_k = len(c)
        v1 = Response_update_method.compensation_return(c[decision_make],o[decision_make],o_[decision_make])
        v2 = Response_update_method.fairness_deviation(c,o,o_,number_k,v1)
        v3 = Response_update_method.cognitive_loss(np.dot(o_,np.array(weight_value).T),o[decision_make])
        decision_make_total_utility = v1*alpha_beta_gamma_lambda[0]+\
                                      v2*alpha_beta_gamma_lambda[1]+\
                                      v3*alpha_beta_gamma_lambda[2]
        return decision_make_total_utility

    @staticmethod
    def compensation_return(ci, oi, o_i):
        f1 = np.copy(ci*((o_i -oi) ** 2))
        return f1

    @staticmethod
    def fairness_deviation(c, o, o_, number_k, f1):
        fi = 0
        for D_index in range(number_k):
            fj = np.copy(Utility_function.compensation_return(c[D_index], o[D_index], o_[D_index]))
            fi += np.absolute(np.array([fj - f1]))[0]
        f2 = fi/(number_k-1)
        return f2

    @staticmethod
    def cognitive_loss(oc_, o_initial):
        f3 = np.copy(((oc_- o_initial)**2))
        return f3