import multiprocessing

import numpy as np
from best_response_update_method import Response_update_method
from multiprocessing.dummy import Pool


class DE_algorithom:
    def __init__(self, iteration_numbers,initial_information,maximal_number_of_iterations,epsilon):

        self.iteration_numbers = iteration_numbers
        self.initial_information = initial_information
        self.maximal_number_of_iterations = maximal_number_of_iterations
        self.epsilon = epsilon

    def training(self, particle_number, mf_l, mf_u, CR):
        iteration_numbers = self.iteration_numbers
        initial_information = self.initial_information
        maximal_number_of_iterations = self.maximal_number_of_iterations
        epsilon = self.epsilon
        # initialization
        particle_storage = {}
        result_storage = []
        modified_opinion = []
        particle_best = [0,0]
        decision_makers_numbers = len(initial_information['decision_makers'])
        for i in range(particle_number):
            particle_storage['%d' % i] = {}
        for i1 in range(particle_number):
            z = (1-epsilon)*(np.random.rand(1)-1.2)+epsilon
            l = (initial_information['original_opinion'] + z * (np.dot(initial_information['original_opinion'], initial_information['weight_value'].T) - initial_information['original_opinion']))
            particle_storage['%d'%i1]['0'] = np.hstack((np.random.rand(decision_makers_numbers)*10,l))
            particle_storage['%d' % i1]['0'][0:decision_makers_numbers] = particle_storage['%d' % i1]['0'][0:decision_makers_numbers].clip(0.0001, 10)
            particle_storage['%d' % i1]['0'][decision_makers_numbers:] = particle_storage['%d' % i1]['0'][decision_makers_numbers:].clip(0, 1)
            judge = True
            while judge:
                particle_storage['%d' % i1]['0'][0:decision_makers_numbers] = particle_storage['%d' % i1]['0'][0:decision_makers_numbers].clip(0.0001, 10)
                particle_storage['%d' % i1]['0'][decision_makers_numbers:] = particle_storage['%d' % i1]['0'][decision_makers_numbers:].clip(0, 1)
                initial_information['unit_cost'] = np.copy(particle_storage['%d' % i1]['0'][0:decision_makers_numbers])
                initial_information['suggested_opinion'] = np.copy(particle_storage['%d' % i1]['0'][decision_makers_numbers:])
                best_response_initial = Response_update_method(initial_information, maximal_number_of_iterations)
                modified_opinion_initial, iteration_k1_initial = best_response_initial.iteration()
                # Calculate two function values
                g_x_initial = epsilon - 1 + np.sum(np.absolute((np.array(modified_opinion_initial) - np.dot(initial_information['weight_value'],np.array(modified_opinion_initial).T)))) / decision_makers_numbers
                h_x_initial = iteration_k1_initial - maximal_number_of_iterations
                # Determine whether the generated particle satisfies conditions
                if g_x_initial <= 0 and h_x_initial <= 0:
                    judge = False
                else:
                    particle_storage['%d' % i1]['0'][0:decision_makers_numbers] = np.copy(particle_storage['%d' % i1]['0'][0:decision_makers_numbers]+(np.random.rand(decision_makers_numbers)-0.25)*2)
                    particle_storage['%d' % i1]['0'][decision_makers_numbers:] = np.copy(particle_storage['%d' % i1]['0'][decision_makers_numbers:]+(np.random.rand(1)-0.5)/5)
        # the stopping criteria for the DE algorithm is a fixed number of generations
        for i2 in range(iteration_numbers):
            print(i2)
            iteration_result = []
            iteration_k1_result = []
            iteration_modified_opinion_result = []
            particle_list = []
            lock = multiprocessing.Lock()
            for list_i in range(particle_number):
                list_single = (list_i,particle_storage['%d'%list_i]['%d'%i2][0:decision_makers_numbers],particle_storage['%d'%list_i]['%d'%i2][decision_makers_numbers:],lock)
                particle_list.append(list_single)
            # Accelerated calculation
            pool = Pool(processes=128)
            result = pool.map(self.Response,particle_list)
            for list_output in range(particle_number):
                iteration_result.append(result[list_output][0])
                iteration_k1_result.append(result[list_output][1])
                iteration_modified_opinion_result.append(result[list_output][2])
            best_result = np.min(iteration_result)
            best_single = np.argmin(iteration_result)
            while True:
                initial_information['unit_cost'] = np.copy(particle_storage['%d' % best_single]['%d'%i2][0:decision_makers_numbers])
                initial_information['suggested_opinion'] = np.copy(particle_storage['%d' % best_single]['%d'%i2][decision_makers_numbers:])
                best_response = Response_update_method(initial_information,maximal_number_of_iterations)
                modified_opinion_best,iteration_k1 = best_response.iteration()
                g_x = epsilon - 1 + np.sum(np.absolute((np.array(modified_opinion_best) - np.dot(initial_information['weight_value'], np.array(modified_opinion_best).T)))) / decision_makers_numbers
                h_x =iteration_k1 - maximal_number_of_iterations
                if g_x <= 0 and h_x <= 0:
                    break
                else:
                    iteration_result[best_single] = 10000
                    best_result = np.min(iteration_result)
                    best_single = np.argmin(iteration_result)
            if i2 ==0:
                result_storage=[best_result]
                particle_best[0] =i2
                particle_best[1] =best_single
                modified_opinion = modified_opinion_best
            else:
                if best_result<np.min(result_storage):
                    result_storage.append(best_result)
                    particle_best[0] = i2
                    particle_best[1] = best_single
                    modified_opinion = modified_opinion_best
                else:
                    result_storage.append(result_storage[i2-1])
            # Renewal particles
            for k in range(particle_number):
                # mutation
                random_particles = []
                # Generate four unequal integers at random
                while True:
                    z = np.random.randint(0,2*decision_makers_numbers,1)[0]
                    if z not in random_particles:
                        if len(random_particles)==0:
                            random_particles=[z]
                        else:
                            random_particles.append(z)
                    if len(random_particles)==4:
                        break
                random_particles_sequence = np.copy(random_particles)
                if iteration_result[random_particles[0]] > iteration_result[random_particles[1]]:
                    random_particles_sequence[0] = random_particles[1]
                    random_particles_sequence[1] = random_particles[0]
                if iteration_result[random_particles[2]] > iteration_result[random_particles[3]]:
                    random_particles_sequence[2] = random_particles[3]
                    random_particles_sequence[3] = random_particles[2]
                d1 = (iteration_result[random_particles_sequence[1]]-iteration_result[best_single])
                d2 = (iteration_result[random_particles_sequence[3]]-iteration_result[best_single])
                if d1 == 0:
                    d1 = 0.0001
                if d2 == 0:
                    d2 = 0.0001
                mf1 = mf_l+(mf_u-mf_l)*(iteration_result[random_particles_sequence[0]]-iteration_result[best_single])/d1
                mf2 = mf_l+(mf_u-mf_l)*(iteration_result[random_particles_sequence[2]]-iteration_result[best_single])/d2
                v_iteration = particle_storage['%d'%best_single]['%d'%i2]+mf1*(particle_storage['%d'%random_particles_sequence[0]]['%d'%i2]-
                                                                               particle_storage['%d'%random_particles_sequence[1]]['%d'%i2])+\
                                                                          mf2*(particle_storage['%d'%random_particles_sequence[2]]['%d'%i2]-
                                                                               particle_storage['%d'%random_particles_sequence[3]]['%d'%i2])
                np.set_printoptions(suppress=True)
                v_iteration[0:decision_makers_numbers] = v_iteration[0:decision_makers_numbers].clip(0.0001, 10)
                v_iteration[decision_makers_numbers:] = v_iteration[decision_makers_numbers:].clip(0, 1)
                # crossover
                i_rand = np.random.randint(0,2*decision_makers_numbers,1)[0]
                u_iteration = np.copy(particle_storage['%d'%k]['%d'%i2])
                for k1 in range(2*decision_makers_numbers):
                    rand_i = np.random.rand(1)[0]
                    if rand_i <= CR or k == i_rand:
                        u_iteration[k1] = np.copy(v_iteration[k1])
                # selection
                g_x = epsilon-1+np.sum(np.absolute((np.array(iteration_modified_opinion_result[k])-np.dot(initial_information['weight_value'],np.array(iteration_modified_opinion_result[k]).T))))/decision_makers_numbers
                h_x = iteration_k1_result[k]-maximal_number_of_iterations
                f_x = iteration_result[k]
                initial_information['unit_cost'] = np.copy(u_iteration[0:decision_makers_numbers])
                initial_information['suggested_opinion'] = np.copy(u_iteration[decision_makers_numbers:])
                best_response_u = Response_update_method(initial_information, maximal_number_of_iterations)
                modified_opinion_u, iteration_k1_u = best_response_u.iteration()
                g_u = epsilon-1+np.sum(np.absolute((np.array(modified_opinion_u)-np.dot(initial_information['weight_value'],np.array(modified_opinion_u).T))))/decision_makers_numbers
                h_u = iteration_k1_u-maximal_number_of_iterations
                f_u = np.dot(initial_information['unit_cost'],np.power(np.array(initial_information['original_opinion']-np.array(modified_opinion_u)),2).T)
                if h_x*h_u == 0:
                    if h_x<=h_u:
                        particle_storage['%d'%k]['%d'%(i2+1)] = np.copy(particle_storage['%d'%k]['%d'%(i2)])
                    else:
                        particle_storage['%d'%k]['%d'%(i2+1)] = np.copy(u_iteration)
                elif h_x*h_u>0:
                    if (g_x<=0 and g_u<=0 and f_u<=f_x)or(g_x>0 and g_u<=0)or(g_x>g_u and g_u>0):
                        particle_storage['%d' % k]['%d' % (i2 + 1)] = np.copy(u_iteration)
                    else:
                        particle_storage['%d' % k]['%d' % (i2 + 1)] = np.copy(particle_storage['%d'%k]['%d'%(i2)])
                particle_storage['%d' % k]['%d' % (i2 + 1)][0:decision_makers_numbers] = particle_storage['%d' % k]['%d' % (i2 + 1)][0:decision_makers_numbers].clip(0.0001, 10)
                particle_storage['%d' % k]['%d' % (i2 + 1)][decision_makers_numbers:] = particle_storage['%d' % k]['%d' % (i2 + 1)][decision_makers_numbers:].clip(0, 1)
        particle_storage['%d' % particle_best[1]]['%d' % particle_best[0]][0:decision_makers_numbers] = particle_storage['%d' % particle_best[1]]['%d' % particle_best[0]][0:decision_makers_numbers].clip(0.0001,10)
        particle_storage['%d' % particle_best[1]]['%d' % particle_best[0]][decision_makers_numbers:] = particle_storage['%d' % particle_best[1]]['%d' % particle_best[0]][decision_makers_numbers:].clip(0,1)
        particle_result = []
        for k_r in range(particle_number):
            particle_result.append(particle_storage['%d' % k_r]['%d' % iteration_numbers])
        return result_storage,modified_opinion,particle_storage['%d'%particle_best[1]]['%d'%(particle_best[0])],particle_result

    # @staticmethod
    def Response(self, particle_list):
        with particle_list[3]:
            initial_information = self.initial_information
            maximal_number_of_iterations = self.maximal_number_of_iterations
            initial_information['unit_cost'] = np.copy(particle_list[1])
            initial_information['suggested_opinion'] = np.copy(particle_list[2])
            best_response = Response_update_method(initial_information, maximal_number_of_iterations)
            modified_opinion, iteration_k1 = best_response.iteration()
            penalty_function = np.dot(initial_information['unit_cost'], np.power( np.array(initial_information['original_opinion'] - np.array(modified_opinion)), 2).T)
        return penalty_function,iteration_k1,modified_opinion









