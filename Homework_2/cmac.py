import numpy as np
import math
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
        """Determine the association vector index to be assigned in the association map for a given input data point.

        Args:
            input_value (float]): Input data value
            num_association_vectors (int): Number of association vectors to generate
            min_input (float): Min input value
            max_input (float): Max input value

        Returns:
            (float): Index to be assigned in the association map
        """
        if input_value < min_input:
            return 1
        elif input_value > max_input:
            return num_association_vectors - 1
        else:
            proportion_idx = (num_association_vectors - 2) * ((input_value - min_input) / (max_input - min_input)) + 1
            return proportion_idx

    def GenerateAssociationMap(self, data, min_input, max_input):
        """Generate the association map dictionary of indexes (value) for each input data point (key).

        Args:
            data (list): List of input data points
            min_input (float): Min input value
            max_input (float): Max input value
        """

        # Calculate number of association vectors to generate
        num_association_vectors = self.num_weights + 1 - self.generalization_factor

        # For each data point (key), determine its association vector index and 
        # assign it to the association map (value)
        for i in range(len(data[0])):
            value = data[0][i]
            association_vec_idx = self.AssocVecIdx(value,
                                                    num_association_vectors, 
                                                    min_input, 
                                                    max_input)
            self.association_map[value] = int(math.floor(association_vec_idx)) # Round index down

    def CalcError(self, predicted, expected):
        """Calculate the error of the model based of the difference in expected and predicted values.

        Args:
            predicted (list): List of predicted outputs from the CMAC model
            expected (list): List of expected outputs predetermined from input data labels

        Returns:
            err_total (float): The percentage of predictions the model was able to correctly determine in decimal.
        """

        err = np.subtract(expected, predicted)
        sum_err_squared = np.sum(np.power(err, 2))
        err_total = np.sqrt(sum_err_squared) / len(expected)

        return err_total

    def Predict(self, data, min_input, max_input, gen_assoc_map = True):
        """Generate predictions using the CMAC model using input data points

        Args:
            data (list): List of input data points
            min_input (float): Min input value
            max_input (float): Max input value
            gen_assoc_map (bool, optional): Determines if the association map will get generated again. Defaults to True.

        Returns:
            predicted (list): List of predicted outputs
            accuracy (float): The percentage of predictions the model was able to correctly determine in decimal.
        """

        predicted = []

        if gen_assoc_map:
            self.GenerateAssociationMap(data, min_input, max_input)

        for i in range(len(data[0])):
            value = [data[0][i], data[1][i]]
            weight_idx = self.association_map[value[0]]

            # Sum the weights in activated cells
            temp_output = np.sum(self.weight_vector[weight_idx : weight_idx + self.generalization_factor])

            predicted.append(temp_output)

        expected = data[:][1]
        err = self.CalcError(predicted, expected)
        accuracy = 1 - err

        return predicted, accuracy

    def TrainModel(self, data, min_input, max_input, epochs = 10000, learning_rate = 0.01):
        """Train the Discrete CMAC model

        Args:
            data (list): List of input data points
            min_input (float): Min input value
            max_input (float): Max input value
            epochs (int, optional): Max number of times the model will iterate in an attempt to converge. Defaults to 10000.
            learning_rate (float, optional): Step value resolution used for correction of weights. Defaults to 0.01.

        Returns:
            time_delta (float): How long the model took to train
        """
        
        # Initialize model parameters
        self.weight_vector = np.ones(self.num_weights)
        current_epoch = 0
        self.association_map = {}
        self.GenerateAssociationMap(data, min_input, max_input)

        prev_err = 0
        current_err = 0
        converged = False
        start_time = time.time()
        while current_epoch <= epochs and not converged:
            prev_err = current_err

            for i in range(len(data[0])):
                value = [data[0][i], data[1][i]]
                # Get index for the beginning of generalization factor window
                weight_idx = self.association_map[value[0]]

                # Output is the sum of the weights within the generalization factor window
                output = np.sum(self.weight_vector[weight_idx : weight_idx + self.generalization_factor])
                err = value[1] - output
                correction = (learning_rate * err) / self.generalization_factor
                
                # Recalculate the weight vector values using the correction coeff
                self.weight_vector[weight_idx : weight_idx + self.generalization_factor] = \
                    [(self.weight_vector[idx] + correction) \
                    for idx in range(weight_idx, (weight_idx + self.generalization_factor))]

           # Calculate the current accuracy of the model 
            _, accuracy = self.Predict(data, min_input, max_input, False)
            current_err = 1 - accuracy

            # Check if the model has congerged
            if np.abs(prev_err - current_err) < 0.0000001:
                converged = True
                        
            current_epoch = current_epoch + 1
        print(f'Discrete CMAC: \n  Generalization Factor: {self.generalization_factor} \n  Epoch: {current_epoch} \n  Percent Error: {current_err * 100}% \n  Accuracy: {accuracy * 100}%')

        end_time = time.time()
        time_delta = end_time - start_time
        return time_delta

class ContinuousCMAC(CMAC):
    def __init__(self, generalization_factor, num_weights):
        CMAC.__init__(self, generalization_factor, num_weights)
    
    def AssocVecIdx(self, input_value, num_association_vectors, min_input, max_input):
        """Determine the association vector index to be assigned in the association map for a given input data point.

        Args:
            input_value (float]): Input data value
            num_association_vectors (int): Number of association vectors to generate
            min_input (float): Min input value
            max_input (float): Max input value

        Returns:
            (float): Index to be assigned in the association map
        """

        if input_value < min_input:
            return 1
        elif input_value > max_input:
            return num_association_vectors - 1
        else:
            proportion_idx = (num_association_vectors - 2) * ((input_value - min_input) / (max_input - min_input)) + 1
            return proportion_idx

    def GenerateAssociationMap(self, data, min_input, max_input):
        """Generate the association map dictionary of indexes (value) for each input data point (key).

        Args:
            data (list): List of input data points
            min_input (float): Min input value
            max_input (float): Max input value
        """
        
        # Calculate number of association vectors to generate
        num_association_vectors = self.num_weights + 1 - self.generalization_factor

        # For each data point (key), determine its association vector index and 
        # assign it to the association map (value)
        for i in range(len(data[0])):
            value = data[0][i]
            association_vec_idx = self.AssocVecIdx(value,
                                        num_association_vectors, 
                                        min_input, 
                                        max_input)
            # Round index up and down
            low_association_vec_idx = int(math.floor(association_vec_idx))
            high_association_vec_idx = int(math.ceil(association_vec_idx))

            if low_association_vec_idx != high_association_vec_idx:
                self.association_map[value] = (low_association_vec_idx, high_association_vec_idx)
            else:
                self.association_map[value] = (low_association_vec_idx, 0)
    
    def CalcError(self, predicted, expected):
        """Calculate the error of the model based of the difference in expected and predicted values.

        Args:
            predicted (list): List of predicted outputs from the CMAC model
            expected (list): List of expected outputs predetermined from input data labels

        Returns:
            err_total (float): The percentage of predictions the model was able to correctly determine in decimal.
        """

        err = np.subtract(expected, predicted)
        sum_err_squared = np.sum(np.power(err, 2))
        err_total = np.sqrt(sum_err_squared) / len(expected)

        return err_total

    def Predict(self, data, min_input, max_input, gen_assoc_map = True):
        """Generate predictions using the Continuous CMAC model using input data points

        Args:
            data (list): List of input data points
            min_input (float): Min input value
            max_input (float): Max input value
            gen_assoc_map (bool, optional): Determines if the association map will get generated again. Defaults to True.

        Returns:
            predicted (list): List of predicted outputs
            accuracy (float): The percentage of predictions the model was able to correctly determine in decimal.
        """

        predicted = []
        inputs = np.linspace(min_input, max_input, self.num_weights + 1 - self.generalization_factor)

        if gen_assoc_map:
            self.GenerateAssociationMap(data, min_input, max_input)

        for i in range(len(data[0])):
            value = [data[0][i], data[1][i]]
            low_association_idx = self.association_map[value[0]][0]
            high_association_idx = self.association_map[value[0]][1]

            l_shared = np.abs(inputs[low_association_idx] - value[0])
            r_shared = np.abs(inputs[high_association_idx] - value[0])

            l_ratio = r_shared / (l_shared + r_shared)
            r_ratio = l_shared / (l_shared + r_shared)

            # Sum the weights in activated cells
            temp_output = (l_ratio * np.sum(self.weight_vector[low_association_idx : low_association_idx + self.generalization_factor])) + (r_ratio * np.sum(self.weight_vector[high_association_idx : high_association_idx + self.generalization_factor]))

            predicted.append(temp_output)

        expected = data[:][1]
        err = self.CalcError(predicted, expected)
        accuracy = 1 - err

        return predicted, accuracy

    def TrainModel(self, data, min_input, max_input, epochs = 10000, learning_rate = 0.01):
        """Train the Continuous CMAC model

        Args:
            data (list): List of input data points
            min_input (float): Min input value
            max_input (float): Max input value
            epochs (int, optional): Max number of times the model will iterate in an attempt to converge. Defaults to 10000.
            learning_rate (float, optional): Step value resolution used for correction of weights. Defaults to 0.01.

        Returns:
            time_delta (float): How long the model took to train
        """

        self.weight_vector = np.ones(self.num_weights)
        current_epoch = 0
        self.association_map = {}
        self.GenerateAssociationMap(data, min_input, max_input)

        prev_err = 0
        current_err = 0
        inputs = np.linspace(min_input, max_input)
        converged = False
        start_time = time.time()
        while current_epoch <= epochs and not converged:
            prev_err = current_err

            for i in range(len(data[0])):
                value = [data[0][i], data[1][i]]
                low_association_idx = self.association_map[value[0]][0]
                high_association_idx = self.association_map[value[0]][1]

                l_shared = np.abs(inputs[low_association_idx] - value[0])
                r_shared = np.abs(inputs[high_association_idx] - value[0])

                l_ratio = r_shared / (l_shared + r_shared)
                r_ratio = l_shared / (l_shared + r_shared)

                output = (l_ratio * np.sum(self.weight_vector[low_association_idx : low_association_idx + self.generalization_factor])) \
                     + (r_ratio * np.sum(self.weight_vector[high_association_idx : high_association_idx + self.generalization_factor]))

                err = value[1] - output
                correction = (learning_rate * err) / self.generalization_factor

                self.weight_vector[low_association_idx : low_association_idx + self.generalization_factor] = \
                    [(self.weight_vector[idx] + correction) for idx in range(low_association_idx, low_association_idx + self.generalization_factor)]
                self.weight_vector[high_association_idx : high_association_idx + self.generalization_factor] = \
                    [(self.weight_vector[idx] + correction) for idx in range(high_association_idx, high_association_idx + self.generalization_factor)]

                _, accuracy = self.Predict(data, min_input, max_input, False)
                current_err = 1 - accuracy
                if np.abs(prev_err - current_err) < 0.0000001:
                    converged = True
                
                current_epoch = current_epoch + 1
        print(f'Continuous CMAC: \n  Generalization Factor: {self.generalization_factor} \n  Epoch: {current_epoch} \n  Percent Error: {current_err * 100}% \n  Accuracy: {accuracy * 100}%')

        end_time = time.time()
        time_delta = end_time - start_time
        return time_delta

