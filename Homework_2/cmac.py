import imp
import numpy as np
import math
import matplotlib
import time

class CMAC:
    def __init__(self, generalization_factor, num_weights):
        self.generalization_factor = generalization_factor
        self.num_weights = num_weights
        self.weight_vector = np.ones(self.num_weights)
        self.association_map = {}

    def SetGenFactor(self, generalization_factor):
        self.generalization_factor = generalization_factor

class DiscreteCMAC(CMAC):
    def __init__(self, generalization_factor, num_weights):
        CMAC.__init__(self, generalization_factor, num_weights)

    def AssocVecIdx(self, input_value, num_association_vectors, min_input, max_input):
        if input_value < min_input:
            return 1
        elif input_value > max_input:
            return num_association_vectors - 1
        else:
            proportion_idx = (num_association_vectors - 2) * ((input_value - min_input) / (max_input - min_input)) + 1
            return proportion_idx

    def GenerateAssociationMap(self, data, min_input, max_input):
        num_association_vectors = self.num_weights + 1 - self.generalization_factor

        for idx, value in data:
            # enumerate()
            association_vec_idx = self.GetAssociationVecIdx(value[0],
                                                            num_association_vectors, 
                                                            min_input, 
                                                            max_input)
            self.association_map[value[0]] = int(math.floor(association_vec_idx)) # Round index down

    def CalcError(self, predicted, expected):
        err = np.subtract(expected, predicted)
        sum_err_squared = np.sum(np.power(err, 2))
        err_total = np.sqrt(sum_err_squared) / len(expected)

        return err_total

    def Predict(self, data, min_input, max_input, gen_assoc_map = True):
        output = []

        if gen_assoc_map:
            self.GenerateAssociationMap(data, min_input, max_input)

        for idx, value in data:
            #enumerate(data)
            weight_idx = self.association_map[value[0]]

            # Sum the weights in activated cells
            temp_output = np.sum(self.weight_vector[weight_idx : weight_idx + self.generalization_factor])

            output.append(temp_output)

        err = self.CalcError(output, data[ : , 1])
        accuracy = 1 - err

        return output, accuracy

    def TrainModel(self, data, min_input, max_input, epochs = 10000, learning_rate = 0.01):
        self.weight_vector = np.ones(self.num_weights)
        # inputs = np.linspace(min_input, max_input)
        current_epoch = 0
        self.association_map = {}
        self.GenerateAssociationMap(data, min_input, max_input)

        prev_err = 0
        current_err = 0
        converged = False
        start_time = time.clock()
        while current_epoch <= epochs and not converged:
            prev_err = current_err
            for idx, value in data:
                #enumerate(data)
                weight_idx = self.association_map[value[0]]
                output = np.sum(self.weight_vector[weight_idx : weight_idx + self.generalization_factor])
                err = value[1] - output
                correction = (learning_rate * err) / self.generalization_factor
                self.weight_vector[weight_idx : weight_idx + self.generalization_factor] = [(self.weight_vector[idx] + correction) for idx in range(weight_idx, (weight_idx + self.generalization_factor))]
            
            _, accuracy = self.Predict(data, min_input, max_input, False)
            current_err = 1 - accuracy

            if np.abs(prev_err, current_err) < 0.0000001:
                converged = True
                        
            current_epoch = current_epoch + 1
            print(f'Discrete CMAC: \n generalization factor: {self.generalization_factor} \n epoch: {current_epoch} \n error: {current_err} \n accuracy: {accuracy * 100}%')

        end_time = time.clock()
        time_delta = end_time - start_time
        return time_delta

class ContinuousCMAC(CMAC):
    def __init__(self, generalization_factor, num_weights):
        CMAC.__init__(generalization_factor, num_weights)
    
    def AssocVecIdx(self, input_value, num_association_vectors, min_input, max_input):
        if input_value < min_input:
            return 1
        elif input_value > max_input:
            return num_association_vectors - 1
        else:
            proportion_idx = (num_association_vectors - 2) * ((input_value - min_input) / (max_input - min_input)) + 1
            return proportion_idx

    def GenerateAssociationMap(self, data, min_input, max_input):
        num_association_vectors = self.num_weights + 1 - self.generalization_factor

        for idx, value in data:
            # enumerate(data)
            association_vec_idx = self.GetAssociationVecIdx(value[0],
                                                            num_association_vectors, 
                                                            min_input, 
                                                            max_input)
            # Round index up and down
            low_association_vec_idx = int(math.floor(association_vec_idx))
            high_association_vec_idx = int(math.ceil(association_vec_idx))

            if low_association_vec_idx != high_association_vec_idx:
                self.association_map[value[0]] = (low_association_vec_idx, high_association_vec_idx)
            else:
                self.association_map[value[0]] = (low_association_vec_idx, 0)
    
    def CalcError(self, predicted, expected):
        err = np.subtract(expected, predicted)
        sum_err_squared = np.sum(np.power(err, 2))
        err_total = np.sqrt(sum_err_squared) / len(expected)

        return err_total

    def Predict(self, data, min_input, max_input, gen_assoc_map = True):
        output = []
        inputs = np.linspace(min_input, max_input, self.num_weights + 1 - self.generalization_factor)

        if gen_assoc_map:
            self.GenerateAssociationMap(data, min_input, max_input)

        for idx, value in data:
            #enumerate(data)
            low_association_idx = self.association_map[value[0][0]]
            high_association_idx = self.association_map[value[0][1]]

            l_shared = np.abs(inputs[low_association_idx] - value[0])
            r_shared = np.abs(inputs[high_association_idx] - value[0])

            l_ratio = r_shared / (l_shared + r_shared)
            r_ratio = l_shared / (l_shared + r_shared)

            # Sum the weights in activated cells
            temp_output = (l_ratio * np.sum(self.weight_vector[low_association_idx : low_association_idx + self.generalization_factor])) + (r_ratio * np.sum(self.weight_vector[high_association_idx : high_association_idx + self.generalization_factor]))

            output.append(temp_output)

        err = self.CalcError(output, data[ : , 1])
        accuracy = 1 - err

        return output, accuracy

    def Train(self, data, min_input, max_input, epochs = 10000, learning_rate = 0.01):
        self.weight_vector = np.ones(self.num_weights)
        current_epoch = 0
        self.association_map = {}
        self.GenerateAssociationMap(data, min_input, max_input)

        prev_err = 0
        current_err = 0
        inputs = np.linspace(min_input, max_input)
        converged = False
        start_time = time.clock()
        while current_epoch <= epochs and not converged:
            prev_err = current_err
            for idx, value in data:
                # enumerate(data)
                low_association_idx = self.association_map[value[0][0]]
                high_association_idx = self.association_map[value[0][1]]

                l_shared = np.abs(inputs[low_association_idx] - value[0])
                r_shared = np.abs(inputs[high_association_idx] - value[0])

                l_ratio = r_shared / (l_shared + r_shared)
                r_ratio = l_shared / (l_shared + r_shared)

                output = (l_ratio * np.sum(self.weight_vector[low_association_idx : low_association_idx + self.generalization_factor])) + (r_ratio * np.sum(self.weight_vector[high_association_idx : high_association_idx + self.generalization_factor]))

                err = value[1] - output
                correction = (learning_rate * err) / self.generalization_factor

                self.weight_vector[low_association_idx : low_association_idx + self.generalization_factor] = [(self.weight_vector(idx) + correction) for idx in range(low_association_idx, low_association_idx + self.generalization_factor)]
                self.weight_vector[high_association_idx : high_association_idx + self.generalization_factor] = [(self.weight_vector[idx] + correction) for idx in range(high_association_idx, high_association_idx + self.generalization_factor)]

                _, accuracy = self.Predict(data, min_input, max_input, False)
                if np.abs(prev_err - current_err) < 0.0000001:
                    converged = True
                
                current_epoch = current_epoch + 1
                print(f'Continuous CMAC: \n generalization factor: {self.generalization_factor} \n epoch: {current_epoch} \n error: {current_err} \n accuracy: {accuracy * 100}%')

            end_time = time.clock()
            time_delta = end_time - start_time
            return time_delta

