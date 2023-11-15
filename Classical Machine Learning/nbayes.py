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
        self.positive_prior = 0
        self.negative_prior = 0
        self.positive_probs = {}
        self.negative_probs = {}


    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema

    # def calculate_mle_params(self):
    #     if m_estimate < 0:
    #


    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        Runs the training algorithm to fit the Naive Bayes classifier to the dataset.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example.
        """

        # Discretize continuous features
        X_discretized = self.discretize_continuous_features(X)

        # Calculate prior probabilities for each class
        self.calculate_priors(y)
        # Calculate the conditional probabilities
        self.calculate_conditional_probabilities(X, y)

    def calculate_priors(self, y: np.ndarray)->None:
        """
        Calculates the class priors for the Naive Bayes model.

        Args:
            y (np.ndarray): The labels. The shape is (n_examples,).

        """
        # Calculate prior probabilities for each class
        [num_neg, num_pos] = util.count_label_occurrences(y)

        self.positive_prior = num_pos / len(y)
        self.negative_prior = num_neg / len(y)


    def calculate_conditional_probabilities(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculates the conditional probabilities for each class.

        Args:
            X: The testing data of shape (n_examples, n_features).
            y (np.ndarray): The labels. The shape is (n_examples,).

        """
        for feature_idx in range(X.shape[1]):
            # Store the examples which have positive labels
            positive_feature_values = X[y == 1, feature_idx]
            # Find the total number of examples in the positive class
            total_pos = len(positive_feature_values)
            # Finds the number of unique values
            pos_counts = util.calculate_value_counts(positive_feature_values)
            self.positive_probs[feature_idx] = {
                value: (count + self.m_estimate) / (total_pos + self.m_estimate * len(pos_counts))
                for value, count in pos_counts.items()
            }

            # Store negative label feature values
            negative_feature_values = X[y == 0, feature_idx]
            # Find the total number of examples in the positive class
            total_neg = len(negative_feature_values)
            # # Finds the number of unique values
            neg_counts = util.calculate_value_counts(negative_feature_values)
            self.negative_probs[feature_idx] = {
                value: (count + self.m_estimate) / (total_neg + self.m_estimate * len(neg_counts))
                for value, count in neg_counts.items()
            }


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluates the Naive Bayes model.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """

        predictions = []
        for instance in X:
            pos_prob = self.positive_prior
            neg_prob = self.negative_prior
            for feature_idx, value in enumerate(instance):
                # Calculate the conditional probability for positive and negative classes
                if feature_idx in self.positive_probs:
                    pos_prob *= self.positive_probs[feature_idx].get(value, 1 / (
                                len(self.positive_probs[feature_idx]) + self.m_estimate))
                if feature_idx in self.negative_probs:
                    neg_prob *= self.negative_probs[feature_idx].get(value, 1 / (
                                len(self.negative_probs[feature_idx]) + self.m_estimate))

            # Choose the class with the highest probability
            if pos_prob >= neg_prob:
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)

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

            if np.issubdtype(X[:, feature_idx].dtype, np.integer):
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
    y_hat = nbayes.predict(X)

    # Calculate accuracy
    acc = util.accuracy(y, y_hat)
    prec = util.precision(y, y_hat)
    rec = util.recall(y, y_hat)

    return acc, prec, rec



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

    if use_cross_validation:
        # Split datasets into 5 folds using cross validation
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    for X_train, y_train, X_test, y_test in datasets:
        n_bayes = NaiveBayes(schema, num_bins, m_estimate)
        n_bayes.fit(X_train, y_train)
        acc, prec, rec = evaluate_and_print_metrics(n_bayes, X_test, y_test)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)

    avg_accuracy = np.mean(accuracies)
    std_deviation_acc = np.std(accuracies)
    avg_precision = np.mean(precisions)
    std_deviation_prec = np.std(precisions)
    avg_recall = np.mean(recalls)
    std_deviation_rec = np.std(recalls)

    print(f"Accuracy: {avg_accuracy:.3f} {std_deviation_acc:.3f}")
    print(f"Precision: {avg_precision:.3f} {std_deviation_prec:.3f}")
    print(f"Recall: {avg_recall:.3f} {std_deviation_rec:.3f}")


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
