import math

import numpy as np

class Utility_function:
    def __init__(self,
                 sk,
                 w,
                 decision_makes_number,
                 alpha_beta_gamma_lambda):
        self.D = sk['Dk']
        self.O = sk['Ok']
        self.Os = sk['Osk']
        self.C = sk['Ck']
        self.alpha_beta_gamma_lambda = alpha_beta_gamma_lambda
        self.decision_makes_number = decision_makes_number
        self.w_k = w

    def total_function(self):
        D = np.array(self.D)
        O = np.array(self.O)
        Os = np.array(self.Os)
        C = np.array(self.C)
        w_k = self.w_k
        decision_makes_number = self.decision_makes_number
        alpha_beta_gamma_lambda = self.alpha_beta_gamma_lambda
        real_data = np.zeros((1, decision_makes_number, 4, Os.shape[0]))-100
        contrast_data_1 = np.zeros((1, decision_makes_number, 3, Os.shape[0]))-100
        contrast_data_2 = np.zeros((1, decision_makes_number, 3, Os.shape[0])) - 100
        t_d = []
        for zq in range(decision_makes_number):
            t_d.append(-1)
        for t_index in range(Os.shape[0]):
            oc_ = np.dot(w_k, O[t_index + 1].T)
            d1 = 0
            for D_index in range(D.shape[0]):
                d1 += Utility_function.compensation_return(1,
                                                             O[t_index][D_index],
                                                             oc_)/D.shape[0]
            for D_index in range(D.shape[0]):
                a1 = Utility_function.compensation_return(C[t_index][D_index],
                                                            O[t_index][D_index],
                                                            O[t_index + 1][D_index])
                b1 = Utility_function.fairness_deviation(C[t_index],
                                                   O[t_index],
                                                   O[t_index + 1],
                                                   D.shape[0],
                                                   a1)
                c1 = Utility_function.cognitive_loss(oc_,
                                                         O[t_index][D_index])
                # complete modification
                if O[t_index + 1][D_index] == Os[t_index][D_index]:
                    a2 = 0
                    o_unchange = np.copy(O[t_index + 1])
                    o_unchange[D_index] = np.copy(O[t_index][D_index])
                    b2 = Utility_function.fairness_deviation(C[t_index],
                                                       O[t_index],
                                                       o_unchange,
                                                       D.shape[0],
                                                       a2)
                    oc__ = np.dot(w_k, o_unchange.T)
                    c2 = Utility_function.cognitive_loss(oc__,
                                                             O[t_index][D_index])
                    if np.dot(np.array([a1, b1, c1]), alpha_beta_gamma_lambda[D_index].T) > np.dot(
                            np.array([a2, b2, c2]), alpha_beta_gamma_lambda[D_index].T):
                        t_d[D_index] += 1
                        real_data[0][D_index][0][t_d[D_index]] = a1
                        real_data[0][D_index][1][t_d[D_index]] = b1
                        real_data[0][D_index][2][t_d[D_index]] = c1
                        real_data[0][D_index][3][t_d[D_index]] = d1
                        contrast_data_1[0][D_index][0][t_d[D_index]] = a2
                        contrast_data_1[0][D_index][1][t_d[D_index]] = b2
                        contrast_data_1[0][D_index][2][t_d[D_index]] = c2
                # No modification
                elif O[t_index + 1][D_index] == O[t_index][D_index]:

                    o_completely_change = np.copy(O[t_index + 1])
                    o_completely_change[D_index] = np.copy(Os[t_index][D_index])
                    a2 = Utility_function.compensation_return(
                        C[t_index][D_index],
                        O[t_index][D_index],
                        Os[t_index][D_index])
                    b2 = Utility_function.fairness_deviation(C[t_index],
                                                          O[t_index],
                                                          o_completely_change,
                                                          D.shape[0],
                                                       a2)
                    oc__ = np.dot(w_k, o_completely_change.T)
                    c2 = Utility_function.cognitive_loss(oc__,
                                                             O[t_index][D_index])
                    if np.dot(np.array([a1, b1, c1]), alpha_beta_gamma_lambda[D_index].T) > np.dot(
                            np.array([a2, b2, c2]), alpha_beta_gamma_lambda[D_index].T):
                        t_d[D_index] += 1
                        real_data[0][D_index][0][t_d[D_index]] = a1
                        real_data[0][D_index][1][t_d[D_index]] = b1
                        real_data[0][D_index][2][t_d[D_index]] = c1
                        real_data[0][D_index][3][t_d[D_index]] = d1
                        contrast_data_1[0][D_index][0][t_d[D_index]] = a2
                        contrast_data_1[0][D_index][1][t_d[D_index]] = b2
                        contrast_data_1[0][D_index][2][t_d[D_index]] = c2
                # Partial modification
                else:
                    a2 = 0
                    o_unchange = np.copy(O[t_index + 1])
                    o_unchange[D_index] = np.copy(O[t_index][D_index])
                    b2 = Utility_function.fairness_deviation(C[t_index],
                                                          O[t_index],
                                                          o_unchange,
                                                          D.shape[0],
                                                       a2)
                    oc__ = np.dot(w_k, o_unchange.T)
                    c2 = Utility_function.cognitive_loss(oc__,
                                                             O[t_index][D_index])
                    o_completely_change = np.copy(O[t_index + 1])
                    o_completely_change[D_index] = np.copy(Os[t_index][D_index])
                    a3 = Utility_function.compensation_return(C[t_index][D_index],
                                                                O[t_index][D_index],
                                                                Os[t_index][D_index])
                    b3 = Utility_function.fairness_deviation(C[t_index],
                                                          O[t_index],
                                                          o_completely_change,
                                                          D.shape[0],
                                                       a3)
                    oc__ = np.dot(w_k, o_completely_change.T)
                    c3 = Utility_function.cognitive_loss(oc__,
                                                             O[t_index][D_index])
                    if np.dot(np.array([a1, b1, c1]), alpha_beta_gamma_lambda[D_index].T) > np.dot(
                            np.array([a2, b2, c2]), alpha_beta_gamma_lambda[D_index].T) and np.dot(
                            np.array([a1, b1, c1]), alpha_beta_gamma_lambda[D_index].T) > np.dot(
                            np.array([a3, b3, c3]), alpha_beta_gamma_lambda[D_index].T):
                        t_d[D_index] += 1
                        real_data[0][D_index][0][t_d[D_index]] = a1
                        real_data[0][D_index][1][t_d[D_index]] = b1
                        real_data[0][D_index][2][t_d[D_index]] = c1
                        real_data[0][D_index][3][t_d[D_index]] = d1
                        contrast_data_1[0][D_index][0][t_d[D_index]] = a2
                        contrast_data_1[0][D_index][1][t_d[D_index]] = b2
                        contrast_data_1[0][D_index][2][t_d[D_index]] = c2
                        contrast_data_2[0][D_index][0][t_d[D_index]] = a3
                        contrast_data_2[0][D_index][1][t_d[D_index]] = b3
                        contrast_data_2[0][D_index][2][t_d[D_index]] = c3
        return real_data, contrast_data_1, contrast_data_2

    @staticmethod
    def compensation_return(ci, oi, o_i):
        f1 = np.copy(ci * ((o_i - oi) ** 2))
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
    def cognitive_loss(oc_, o_):
        f3 = np.copy(( (oc_ - o_) ** 2))
        return f3
