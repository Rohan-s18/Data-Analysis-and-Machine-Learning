"""
Source Code for Logistic Regression Classifier
"""

# Imports
import numpy as np
import argparse
import os.path
import warnings

import warnings

#suppress warnings
warnings.filterwarnings('ignore')

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45

import util

# Class for Logistic regression classifier
class LogisticRegression(Classifier):

    # Intial Attributes
    def __init__(self, schema: List[Feature], lambda_):
        """
        Args:
            lambda_penalty terms
        Return:
            None
        """
        self.learning_rate = 0.01
        self.num_iterations = 1000
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self._schema = schema


    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema


    # Sigmoid Activation function
    def sigmoid(self, z):
        """
        Args:
            z (float)
        Returns:
            sigma(z)
        """
        return 1 / (1 + np.exp(-z))
    

    # Function to replace nominal features 1-of-N encoded vectors
    def one_hot_encode(data, nominal_column_index):
        """
        Args:
            data: Features
            column_index: the index which holds the nominal attributes
        Returns:
            data_encoded: data with the replaced one-hot-encoded vectors for the nominal features
        """

        # Extracting the nominal column
        nominal_column = data[:, nominal_column_index]

        # Getting the unique values in the nominal column
        unique_values = np.unique(nominal_column)

        # Creating an identity matrix with the number of unique values
        identity_matrix = np.eye(len(unique_values))

        # Creating a dictionary to map unique values to their corresponding one-hot encoding
        encoding_dict = dict(zip(unique_values, identity_matrix))

        # Mapping each element in the nominal column to its one-hot encoding
        one_hot_encoded = np.array([encoding_dict[val] for val in nominal_column])

        # Concatenating the one-hot encoded columns to the original data
        data_encoded = np.concatenate([data, one_hot_encoded], axis=1)

        # Removing the original nominal column
        data_encoded = np.delete(data_encoded, nominal_column_index, axis=1)

        return data_encoded


    # Training cycle
    def fit(self, X, y):
        """
        Args:
            X: Input Data
            y: target vector
        Returns:
            none
        """


        m, n = X.shape  # Number of samples and features
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.num_iterations):
            # Computing the logits

            z = np.dot(X, self.weights) + self.bias

            # Applying the sigmoid function
            predictions = self.sigmoid(z)

            # Computing the gradients
            penalty_term = (self.lambda_ /2) * np.linalg.norm(self.weights)
            dw = (1 / m) * np.dot(X.T, (predictions - y)) + penalty_term
            db = (1 / m) * np.sum(predictions - y)

            # Updating the weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    # Predction function
    def predict(self, X):
        """
        Args: 
            X: data variable
        Return:
            predictions: vector of the predictions based on the data values
        """

        #z = np.dot(X, self.weights) + self.bias
        predictions = np.array([(self.sigmoid(np.dot(x_elem, self.weights) + self.bias) > 0.5).astype(int) for x_elem in X])
        
        #return (predictions > 0.5).astype(int)
        return predictions

    
    # Function to return the y-scores
    def y_scores(self, X):
        """
        Args:
            X: data variable
        Return:
            y_scores: vector of the y scores
        """
        y_scores = np.array([self.sigmoid(np.dot(x_elem, self.weights) + self.bias) for x_elem in X])

        return y_scores

    

    



# Function to use the logistic regression classifier
def logreg(data_path: str, lambda_: float, use_cross_validation: bool = True):
    """
    Runs the logistic regression algorithm on the specified dataset.

    Args:
        data_path (str): The path to the data.
        use_cross_validation (bool, optional): If True, use cross validation. Otherwise, run on the full dataset.
        lambda: constant (λ) times a penalty term, half of the 2-norm of the weights square.

    Returns:
        None
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    # List to store the metrics for each fold
    accuracy_list = []
    precision_list = []
    recall_list = []
    #roc_auc_list = []
    all_y_test = []
    confidence_values = []

    i = 1

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)


    for X_train, y_train, X_test, y_test in datasets:

        regressor = LogisticRegression(schema=schema,lambda_=lambda_)
        regressor.fit(X_train, y_train)

        y_hat = regressor.predict(X=X_test)
        y_scores = regressor.y_scores(X=X_test)

        # Evaluating the performance metrics
        accuracy_list.append(util.accuracy(y=y_test, y_hat=y_hat))
        precision_list.append(util.precision(y=y_test, y_hat=y_hat))
        recall_list.append(util.recall(y=y_test,y_hat=y_hat))

        all_y_test.extend(y_test)
        confidence_values.extend(y_scores)

    # Calculating the averages and standard deviations
    accuracy_avg = np.mean(accuracy_list)
    precision_avg = np.mean(precision_list)
    recall_avg = np.mean(recall_list)

    accuracy_std = np.std(accuracy_list)
    precision_std = np.std(precision_list)
    recall_std = np.std(recall_list)

    auc = util.auc(np.array(all_y_test), np.array(confidence_values))

    # Printing the metrics
    print(f"Accuracy: {accuracy_avg:.3f} {accuracy_std:.3f}")
    print(f"Precision: {precision_avg:.3f} {precision_std:.3f}")
    print(f"Recall: {recall_avg:.3f} {recall_std:.3f}")
    print(f"Area Under ROC: {auc:.3f}")

    return [accuracy_avg, accuracy_std, precision_avg, precision_std, recall_avg, recall_std, auc] 




if __name__ == "__main__":
    # Setting up arguments

    parser = argparse.ArgumentParser(description="Run a Logistic Regression algorithm.")

    parser.add_argument("path", metavar="PATH", type=str, 
                        help="The path to the data.")

    parser.add_argument("--no-cv", dest="cv", action="store_false", 
                        help="Disables cross validation and trains on the full dataset.")

    parser.add_argument("lambda_",metavar="lambda_",type=float, 
                        help=" constant (λ) times a penalty term, half of the 2-norm of the weights square.")
    
    parser.set_defaults(cv=True,lambda_=0.0)
    args = parser.parse_args()

    # If the lambda limit is negative throw an exception
    if args.lambda_ < 0:
        raise argparse.ArgumentTypeError("λ must be non-negative.")

    # Storing the args in variables
    data_path = os.path.expanduser(args.path)
    use_cross_validation = args.cv
    lambda_ = args.lambda_

    # Creating a logistic regression classifier with the necessary parameters
    logreg(data_path=data_path, lambda_=lambda_, use_cross_validation=use_cross_validation)


