import cmac
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib

def Hw2Q2(train_x, train_y, min_input, max_input, generalization_factor, num_weights):
    d_cmac = cmac.DiscreteCMAC(generalization_factor, num_weights)
    data = [train_x, train_y]
    d_cmac.TrainModel(data, min_input, max_input)
    return d_cmac

def Hw2Q3(train_x, train_y, min_input, max_input, generalization_factor, num_weights):
    c_cmac = cmac.ContinuousCMAC(generalization_factor, num_weights)
    data = [train_x, train_y]
    c_cmac.TrainModel(data, min_input, max_input)
    return c_cmac

if __name__ == '__main__':
    
    # Generate 100 data points and lables
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    min_input = np.min(x)
    max_input = np.max(x)

    # Split data into train (70) and test (30) subsets
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
    
    generalization_factor = 10 
    num_weights = 35

    # Train Discrete CMAC
    d_cmac = Hw2Q2(train_x, train_y, min_input, max_input, generalization_factor, num_weights)

    # Test Discrete CMAC


    # Train Continuous CMAC
    c_cmac = Hw2Q3(train_x, train_y, min_input, max_input, generalization_factor, num_weights)

    # Test Continuous CMAC

