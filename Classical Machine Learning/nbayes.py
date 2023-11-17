import argparse
import os.path
import warnings

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45

import util


class NaiveBayes(Classifier):
    def __init__(self, schema: List[Feature], num_bins, m_estimate):

        self._schema = schema
        self.num_bins = num_bins
        self.m_estimate = m_estimate
        self.priors = []
        self.conditional_probs = {}


    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema


    def calculate_estimate_params(self, v):
        """
        Calculates the parameters for m-estimates.

            Args:
                v: The number of unique feature values for a specific feature.
            Returns:
                m (int): The m-estimate parameter.
                p (int): The p parameter.
        """
        if self.m_estimate < 0:
            # Use Laplace Smoothing
            m= v
            p = 1/v

        else:
            m = self.m_estimate
            p = v

        return m, p

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        Runs the training algorithm to fit the Naive Bayes classifier to the dataset.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example.
        """

        # Calculate prior probabilities for each class
        self.calculate_priors(y)
        # Calculate the conditional probabilities
        self.conditional_probs = self.calculate_conditional_probabilities(X, y)

    def calculate_priors(self, y: np.ndarray)->None:
        """
        Calculates the class priors for the Naive Bayes model.

        Args:
            y (np.ndarray): The labels. The shape is (n_examples,).

        """
        # Calculate prior probabilities for each class
        [num_neg, num_pos] = util.count_label_occurrences(y)

        pos_prior = num_pos / len(y)
        neg_prior = num_neg / len(y)
        self.priors = [neg_prior, pos_prior]

    def calculate_conditional_probabilities(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Calculates the conditional probabilities for each class.

        Args:
            X: The testing data of shape (n_examples, n_features).
            y (np.ndarray): The labels. The shape is (n_examples,).

        Returns:
            A dictionary of conditional probabilities for each feature.

        """
        conditional_probs = {}

        # Loop through each feature
        for feature_idx in range(X.shape[1]):
            # Get every unique value of the feature
            unique_feat_vals = np.unique(X[:, feature_idx])

            # Get parameters for m-estimates
            m, p = self.calculate_estimate_params(len(unique_feat_vals))

            # Dictionary to store all conditional probabilities for that feature
            conditional_probs[feature_idx] = {}

            # Loop through each class in y
            for class_label in np.unique(y):
                # Examples from X where y = the class label
                X_class = X[y == class_label]

                unique_feature_vals_class, num_vals = np.unique(X_class[:, feature_idx], return_counts=True)
                num_feature_vals = len(unique_feature_vals_class)

                # Dictionary to store each unique feature value and its count
                feature_vals_counts = {value: count for value, count in zip(unique_feature_vals_class, num_vals)}

                # Find the total number of examples that have that class label
                total_examples = len(X_class)

                # Calculate the conditional probability for each unique feature value
                for value in unique_feat_vals:
                    count_val = feature_vals_counts.get(value, 0)
                    conditional_prob = (count_val + (m * p))/(total_examples + m)

                    conditional_probs[feature_idx][(value, class_label)] = conditional_prob

        return conditional_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluates the Naive Bayes model.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """

        predictions = []
        confidence_values= []
        # Loop through examples in X
        for x in X:
            max_prob = float('-inf')
            predicted_class = None
            class_probs = []

            # Loop through each class label
            for class_label in [0,1]:
                class_prob = self.priors[class_label]  # Class prior for the current sample

                # Calculate the product of conditional probabilities for each feature of that example
                for feature_idx, value in enumerate(x):
                    # Get conditional probability for the current feature value given the class
                    conditional_prob = self.conditional_probs.get(feature_idx, {}).get((value, class_label), 0)

                    # Multiply conditional probabilities
                    class_prob *= conditional_prob

                class_probs.append(class_prob)

                # Update predicted class if the current class probability is higher
                if class_prob > max_prob:
                    max_prob = class_prob
                    predicted_class = class_label

            # Append the predicted class to the list of predictions
            predictions.append(predicted_class)

            # Compute confidence intervals as the ratio between class probabilities
            confidence = class_probs[1] / sum(class_probs)
            confidence_values.append(confidence)

        return np.array(predictions), confidence_values

    def discretize_continuous_features(self, X: np.ndarray) -> np.ndarray:
        """
        Discretize continuous features.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features)
            num_bins (int): Number of bins used to discretize continuous values.

        Returns:
            X_discretized (np.ndarray): The discretized input data of shape (n_samples, n_features).
        """

        num_bins = self.num_bins
        X_discretized = np.zeros_like(X, dtype=int)

        # Grab the data type of the feature at the current index
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]

            if np.issubdtype(X[:, feature_idx].dtype, np.integer) or X[:, feature_idx].dtype == object:
                # If the feature is already discrete, leave as is
                X_discretized[:, feature_idx] = feature_values
            else:
                # Discretize continuous features
                min_value = np.min(feature_values)
                max_value = np.max(feature_values)
                bin_boundaries = np.linspace(min_value, max_value, num_bins + 1)

                for i in range(1, num_bins + 1):
                    # Map the original features within the current bins range to that bin
                    X_discretized[(feature_values >= bin_boundaries[i - 1]) & (
                            feature_values <= bin_boundaries[i]), feature_idx] = i

        return X_discretized


def evaluate_and_print_metrics(nbayes: NaiveBayes, X: np.ndarray, y: np.ndarray):
    """
    Evaluates the Naive Bayes Classifier and prints various metrics.

    Args:
        nbayes (NaiveBayes): The Naive Bayes classifier to be evaluated.
        X (np.ndarray): The dataset. The shape is (n_examples, n_features).
        y (np.ndarray): The labels. The shape is (n_examples,).

    Note: Prints metrics but does not return any value.

    """
    # Calculate predicted labels
    y_hat, p_y_hat = nbayes.predict(X)

    # Calculate accuracy
    acc = util.accuracy(y, y_hat)
    prec = util.precision(y, y_hat)
    rec = util.recall(y, y_hat)

    return acc, prec, rec, p_y_hat



def nbayes(data_path: str, num_bins: int, m_estimate: int, use_cross_validation: bool = True):
    """
    :param data_path: The path to the data.
    :param num_bins:
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param m_estimate:
    :return:
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    accuracies = []
    precisions = []
    recalls = []
    all_y_test = []
    confidence_values = []

    if use_cross_validation:
        # Split datasets into 5 folds using cross validation
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    for X_train, y_train, X_test, y_test in datasets:
        n_bayes = NaiveBayes(schema, num_bins, m_estimate)

        # Discretize continuous features
        X_train_discretized = n_bayes.discretize_continuous_features(X_train)
        X_test_discretized = n_bayes.discretize_continuous_features(X_test)

        n_bayes.fit(X_train_discretized, y_train)

        all_y_test.extend(y_test)

        acc, prec, rec, confidence_vals = evaluate_and_print_metrics(n_bayes, X_test_discretized, y_test)
        confidence_values.extend(confidence_vals)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)

    auc = util.auc(np.array(all_y_test), np.array(confidence_values))

    avg_accuracy = np.mean(accuracies)
    std_deviation_acc = np.std(accuracies)
    avg_precision = np.mean(precisions)
    std_deviation_prec = np.std(precisions)
    avg_recall = np.mean(recalls)
    std_deviation_rec = np.std(recalls)

    print(f"Accuracy: {avg_accuracy:.3f} {std_deviation_acc:.3f}")
    print(f"Precision: {avg_precision:.3f} {std_deviation_prec:.3f}")
    print(f"Recall: {avg_recall:.3f} {std_deviation_rec:.3f}")
    print(f"Area Under ROC: {auc:.3f}")

    return [avg_accuracy, std_deviation_acc, avg_precision, std_deviation_prec, avg_recall, std_deviation_rec, auc]


if __name__ == '__main__':
    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('num_bins', metavar='BINS', type=int,
                        help='Number of bins used to discretize continuous values. Must be a positive integer greater than 1.')
    parser.add_argument('m_estimate', metavar='M-ESTIMATE', type=int,
                        help='A non-negative integer m used to calculate the m-estimate.m < 0 will use Laplace Smoothing.')
    parser.set_defaults(cv=True)
    args = parser.parse_args()

    # If the number of bins is less than 2 throw an exception
    if args.num_bins < 2:
        raise argparse.ArgumentTypeError('The number of bins must be greater than 1.')

    data_path = os.path.expanduser(args.path)
    num_bins = args.num_bins
    use_cross_validation = args.cv
    m_estimate = args.m_estimate

    nbayes(data_path, num_bins, m_estimate, use_cross_validation)
