import numpy as np

from data_generation import Data_generated
from comparison import Data_processing
import copy


class Particle_swarm_optimization:
    def __init__(self,
                 initial_data_generated,
                 boundary_value,
                 C1,
                 C2,
                 defined_utility_function_parameters):
        self.initial_data_generated = initial_data_generated
        self.boundary_value = boundary_value
        self.C1 = C1
        self.C2 = C2
        self.defined_utility_function_parameters = defined_utility_function_parameters

    def get_and_iteration_particle(self,c1,c2,c3,particle_number,iteration_number):
        initial_data_generated = self.initial_data_generated
        decision_makers = np.arange(1,16).tolist()
        C1 = self.C1
        C2 = self.C2
        defined_utility_function_parameters = self.defined_utility_function_parameters
        # weight_value = initial_data_generated['1']['weight_value']
        particle_storage = {}
        velocity_storage = {}
        ff_storage = {}
        initial_and_result_single = {}
        k1 = [1,2]
        total2 = {}
        total1 = []
        ff_best = []
        particle_best = {}
        for single_index in range(len(decision_makers)):
            particle_storage['%d'%(single_index+1)] = {}
            velocity_storage['%d'%(single_index+1)] = {}
            ff_storage['%d'%(single_index+1)] = {}
            initial_and_result_single['%d'%(single_index+1)] = {}
            total2['%d'%(single_index+1)] = []
        l_i,li,S = Particle_swarm_optimization.initial_and_result_data_generated(initial_data_generated)
        for single_index in range(len(decision_makers)):
            kl = defined_utility_function_parameters
            for particle_index in range(particle_number):
                a = np.copy(np.random.rand(3).round(4))
                a[1] = np.copy(-a[1])
                a[2] = np.copy(-a[2])
                b = np.copy(np.random.rand(3).round(4))
                particle_storage['%d'%(single_index+1)]['%d' % (particle_index + 1)] = {'0': a}
                velocity_storage['%d'%(single_index+1)]['%d' % (particle_index + 1)] = {'0': b / 10}
            best = 0
            for iteration_index in range(iteration_number):
                print(iteration_index)
                c1 = c1+(1-c1)*(iteration_number-iteration_index-1)/iteration_number
                result_storage = {}
                result_total = []
                best = 0
                for k in range(particle_number):
                    result1,result2 = (Particle_swarm_optimization.function_optimization(l_i, li['%d' % (single_index + 1)],single_index + 1,particle_storage['%d' % (single_index + 1)]['%d' % (k + 1)]['%d' % (iteration_index)],C1,C2))
                    if iteration_index == 0:
                        initial_and_result_single['%d' % (single_index + 1)]['%d' % (k + 1)] = [result1]
                        ff_storage['%d' % (single_index + 1)]['%d' % (k + 1)] = [result2]
                    else:
                        initial_and_result_single['%d' % (single_index + 1)]['%d' % (k + 1)].append(result1)
                        ff_storage['%d' % (single_index + 1)]['%d' % (k + 1)].append(result2)
                    result_total.append(initial_and_result_single['%d'%(single_index+1)]['%d'%(k+1)][iteration_index])

                best = np.argmin(result_total)
                best_single = result_total[best]
                if iteration_index==0:
                    total2['%d' % (single_index + 1)].append(best_single)
                    particle_best['%d'%(single_index+1)] = [best+1,iteration_index]
                else:
                    if best_single<np.min(total2['%d'%(single_index+1)]):
                        total2['%d'%(single_index+1)].append(best_single)
                        particle_best['%d' % (single_index + 1)] = [ best + 1,iteration_index]
                    else:
                        total2['%d'%(single_index+1)].append(np.min(total2['%d'%(single_index+1)]))
                for k in range(particle_number):
                    best_single = np.argmin(initial_and_result_single['%d'%(single_index+1)]['%d' % (k + 1)])
                    r1 = np.random.rand(1)
                    r2 = np.random.rand(1)
                    velocity_storage['%d' % (single_index + 1)]['%d'%(k+1)]['%d'%(iteration_index+1)] = \
                    c1*np.copy(velocity_storage['%d' % (single_index + 1)]['%d' % (k+1)]['%d'%(iteration_index)])+\
                    c2*r1*np.copy(particle_storage['%d'%(single_index+1)]['%d'%(k+1)]['%d'%(best_single)]-particle_storage['%d' % (single_index + 1)]['%d' % (k+1)]['%d'%(iteration_index)])+\
                    c3*r2*np.copy(particle_storage['%d'%(single_index+1)]['%d'%particle_best['%d' % (single_index + 1)][0]]['%d'%particle_best['%d' % (single_index + 1)][1]]-particle_storage['%d' % (single_index + 1)]['%d' % (k+1)]['%d'%(iteration_index)])

                    velocity_storage['%d' % (single_index + 1)]['%d' % (k + 1)]['%d' % (iteration_index + 1)] = velocity_storage['%d' % (single_index + 1)]['%d'%(k+1)]['%d'%(iteration_index+1)].clip(-0.5, 0.5)
                    particle_storage['%d'%(single_index+1)]['%d' % (k + 1)]['%d'%(iteration_index+1)] = np.copy(particle_storage['%d' % (single_index + 1)]['%d' % (k+1)]['%d'%iteration_index]+\
                                                                               velocity_storage['%d' % (single_index + 1)]['%d'%(k+1)]['%d'%(iteration_index+1)])
                    particle_storage['%d' % (single_index + 1)]['%d' % (k + 1)]['%d' % (iteration_index+1)][1:] = particle_storage['%d' % (single_index + 1)]['%d' % (k + 1)]['%d' % (iteration_index+1)][1:].clip(-1,0)
                    particle_storage['%d' % (single_index + 1)]['%d' % (k + 1)]['%d' % (iteration_index+1)][0] = particle_storage['%d' % (single_index + 1)]['%d' % (k + 1)]['%d' % (iteration_index+1)][0].clip(0,1)
            if single_index ==0:
                total1 = [particle_storage['%d'%(single_index+1)]['%d'%particle_best['%d'%(single_index+1)][0]]['%d'%particle_best['%d'%(single_index+1)][1]]]
                ff_best = [ff_storage['%d'%(single_index+1)]['%d'%particle_best['%d'%(single_index+1)][0]][particle_best['%d'%(single_index+1)][1]]]
            else:
                total1.append(particle_storage['%d'%(single_index+1)]['%d'%particle_best['%d'%(single_index+1)][0]]['%d'%particle_best['%d'%(single_index+1)][1]])
                ff_best.append(ff_storage['%d'%(single_index+1)]['%d'%particle_best['%d'%(single_index+1)][0]][particle_best['%d'%(single_index+1)][1]])
        return total1,total2,ff_best,S

    @staticmethod
    def initial_and_result_data_generated(initial_data_generated):
        data_modified = {}
        d = copy.deepcopy(initial_data_generated)
        data_generated = Data_generated(d, 20)
        # Simulation data generation
        data_t_modified, data_t_suggested, cost = data_generated.data_generated(0.95)
        for z in range(len(d)):
            data_modified['s%d' % (z + 1)] = {}
            data_modified['s%d' % (z + 1)]['Dk'] = np.copy(d['%d' % (z + 1)]['decision_makers'])
            data_modified['s%d' % (z + 1)]['Ok'] = np.copy(data_t_modified['%d' % (z + 1)])
            data_modified['s%d' % (z + 1)]['Osk'] = np.copy(data_t_suggested['%d' % (z + 1)])
            data_modified['s%d' % (z + 1)]['Ck'] = np.copy(cost['%d' % (z + 1)])
        data_contrast = Data_processing(data_modified, d)
        # Generate utility data corresponding to preference rank set
        real_data_total, contrast_data_1_total, contrast_data_2_total, l_i,li = data_contrast.data_processing()
        return l_i,li,data_modified

    @staticmethod
    def function_optimization(l_i,li,x,swarm,C1,C2):
        np.set_printoptions(suppress=True)
        function_optimization1 = 0
        function_optimization2 = 0
        df1 = np.dot(swarm, np.array(l_i['real_data']['%d'%x][0:3,:]))
        df2 = np.dot(swarm, np.array(l_i['contrast_data_1']['%d'%x]))
        df = df1 - df2
        df = df[1:]
        for i in df:
            if i > 0:
                function_optimization1 += 1
        function_optimization2 += np.sum(df)
        df3 = np.dot(swarm, np.array(l_i['real_data']['%d'%x][0:3,:]))
        df4 = np.dot(swarm, np.array(l_i['contrast_data_2']['%d'%x]))
        df_ = df3 - df4
        df_ = df_[1:]
        for i_, j_ in enumerate(df_):
            if l_i['contrast_data_2']['%d' % x][0][i_+1] >= 0:
                if j_ > 0:
                    function_optimization1 += 1
                li += 1
                function_optimization2 += j_

        d = (function_optimization1+C1*function_optimization2)/li
        df1 = function_optimization1/li
        df2 = function_optimization2/li
        function_optimization_d = -d+C2*np.sum(np.power(np.array(swarm),2))
        return function_optimization_d, [df1, df2]
