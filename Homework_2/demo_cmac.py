import cmac
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib

def Hw2Q2(train_x, train_y, test_x, test_y, min_input, max_input, generalization_factor, num_weights):
    """Program a Discrete CMAC and train it on a 1-D function (ref: Albus 1975, Fig. 5) 
    Explore effect of overlap area on generalization and time to convergence. 
    Use only 35 weights for your CMAC, and sample your function at 100 evenly spaced points. 
    Use 70 for training and 30 for testing. Report the accuracy of your CMAC network using 
    only the 30 test points.

    Args:
        train_x (list): Training input data
        train_y (list): Training output data
        test_x (list): Testing input data
        test_y (list): Testing output data
        min_input (float): Lowest input value
        max_input (float): Largest input value
        generalization_factor (int): Generalization factor
        num_weights (int): Number of weights to generate

    Returns:
        class: Trained discrete CMAC model
        list: Predicted outputs using the test data inputs
        float: The percentage of predictions the model was able to correctly determine in decimal
    """

    d_cmac = cmac.DiscreteCMAC(generalization_factor, num_weights)
    data = [train_x, train_y]
    d_cmac.TrainModel(data, min_input, max_input)
    
    data = [test_x, test_y]
    predicted, accuracy = d_cmac.Predict(data, min_input, max_input)
    return d_cmac, predicted, accuracy

def Hw2Q3(train_x, train_y, test_x, test_y, min_input, max_input, generalization_factor, num_weights):
    """ Program a Continuous CMAC by allowing partial cell overlap, and modifying the 
    weight update rule accordingly. Use only 35 weights for your CMAC, and sample your 
    function at 100 evenly spaced points.  Use 70 for training and 30 for testing. 
    Report the accuracy of your CMAC network using only the 30 test points. Compare the 
    output of the Discrete CMAC with that of the Continuous CMAC. 
    (You may need to provide a graph to compare)

    Args:
        train_x (list): Training input data
        train_y (list): Training output data
        test_x (list): Testing input data
        test_y (list): Testing output data
        min_input (float): Lowest input value
        max_input (float): Largest input value
        generalization_factor (int): Generalization factor
        num_weights (int): Number of weights to generate

    Returns:
        class: Trained continuous CMAC model
        list: Predicted outputs using the test data inputs
        float: The percentage of predictions the model was able to correctly determine in decimal
    """

    c_cmac = cmac.ContinuousCMAC(generalization_factor, num_weights)
    data = [train_x, train_y]
    c_cmac.TrainModel(data, min_input, max_input)

    data = [test_x, test_y]
    predicted, accuracy = c_cmac.Predict(data, min_input, max_input)
    return c_cmac, predicted, accuracy

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

    # Train and Test Discrete CMAC
    d_cmac, predicted, accuracy = Hw2Q2(train_x, train_y, test_x, test_y, min_input, max_input, generalization_factor, num_weights)



    # Train and Test Continuous CMAC
    c_cmac, predicted, accuracy = Hw2Q3(train_x, train_y, test_x, test_y, min_input, max_input, generalization_factor, num_weights)
