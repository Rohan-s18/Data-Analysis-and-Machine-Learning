import random
from typing import Tuple, Iterable

import numpy as np
import math


def count_label_occurrences(y: np.ndarray) -> Tuple[int, int]:
    """
    Takes an array of labels and counts the number of positive and negative labels.

    HINT: Maybe a method like this is useful for calculating more complicated things like entropy!

    Args:
        y: Array of binary labels.

    Returns: A tuple containing the number of negative occurrences, and number of positive occurrences, respectively.

    """
    n_ones = np.sum(y == 1)
    n_zeros = y.size - n_ones
    return n_zeros, n_ones


def calculate_value_counts(values):
    """
       Calculates the number of unique values in an array.

       Args:
           values (list): A list representing different feature values.

       Returns:
           value_counts: The number of unique values.

       """
    value_counts = {}
    for val in values:
        if val not in value_counts:
            value_counts[val] = 1
        else:
            value_counts[val] += 1
    return value_counts


def entropy(values_cnt):
    """
    Calculates the entropy of a set of values.

    Args:
        values_cnt (list): A list representing the count of occurrences for each possible value.

    Returns:
        entropy: The calculated entropy.

    """
    n = sum(values_cnt)
    entropy = 0

    # Return entropy = 0 if there are no occurrences of that label
    if n == 0:
        return 0

    # for each value, calculate entropy
    for val in values_cnt:
        p_val = val/n

        if p_val != 0 :
            entropy = entropy - (p_val * math.log2(p_val))
        # Handle the case when p_val is zero
        else:
            entropy = entropy  # Assign zero entropy contribution

    return entropy


def cv_split(
        X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Conducts a cross-validation split on the given data.

    Args:
        X: Data of shape (n_examples, n_features)
        y: Labels of shape (n_examples,)
        folds: Number of CV folds
        stratified:

    Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
    for each fold.

    For example, 5 fold cross validation would return the following:
    (
        (X_train_1, y_train_1, X_test_1, y_test_1),
        (X_train_2, y_train_2, X_test_2, y_test_2),
        (X_train_3, y_train_3, X_test_3, y_test_3),
        (X_train_4, y_train_4, X_test_4, y_test_4),
        (X_train_5, y_train_5, X_test_5, y_test_5)
    )

    """

    # Create a structure to store each training and testing set
    fold_datasets = []

    # Set the RNG seed to 12345 to ensure repeatability
    np.random.seed(12345)
    random.seed(12345)

    # Find the proportion of positive and negative labels if stratified
    if stratified:
        # Find the number of positive and negative labels
        n_zeros, n_ones = count_label_occurrences(y)

        # Find the indices where positive and negative labels occur
        indices_of_ones = np.where(y == 1)[0]
        indices_of_zeros = np.where(y == 0)[0]

        # Calculate number of each label per fold
        n_ones_in_fold = n_ones // folds
        n_zeros_in_fold = n_zeros // folds

        # Loop to create each fold
        for i in range(folds):
            # Randomly shuffle the indices
            np.random.shuffle(indices_of_ones)
            np.random.shuffle(indices_of_zeros)

            # Calculate which indices will be contained in this fold
            ones_fold_indices = indices_of_ones[i * n_ones_in_fold : (i+1)* n_ones_in_fold]
            zeros_fold_indices = indices_of_zeros[i * n_zeros_in_fold : (i+1)* n_zeros_in_fold]
            indices_in_fold = np.concatenate([ones_fold_indices, zeros_fold_indices])

            # Shuffle the indices we will use for this fold
            np.random.shuffle(indices_in_fold)

            # Build the training sets based on the indices we will include in this fold
            X_training = X[indices_in_fold]
            y_training = y[indices_in_fold]

            # Build testing set by deleting data points we already used so the remaining points act as testing datasets
            X_testing = np.delete(X, indices_in_fold, axis=0)
            y_testing = np.delete(y, indices_in_fold)

            fold_datasets.append((X_training, y_training, X_testing, y_testing))

    else:
        # If not stratified, use regular n-fold

        # Calculate the size of each fold by dividing number of rows by number of folds
        fold_size = len(X) // folds

        for i in range(folds):
            # Create an array of indices for the data points in this fold
            indices_in_fold = np.arange(i * fold_size, (i + 1) * fold_size)

            # Build the training sets based on the indices we will include in this fold
            X_training = X[indices_in_fold]
            y_training = y[indices_in_fold]

            # Build testing set by deleting data points we already used so the remaining points act as testing datasets
            X_testing = np.delete(X, indices_in_fold, axis=0)
            y_testing = np.delete(y, indices_in_fold)

            fold_datasets.append((X_training, y_training, X_testing, y_testing))

    return tuple(fold_datasets)


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates the accuracy of the classifier, or the number of correctly predicted positive and negative
        labels divided by the total number of labels

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy
    """

    # If predicted labels and true labels are not the same size, throw an error
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size.')
    # If datasets are empty, throw an error
    if len(y) == 0 or len(y_hat) == 0:
        raise ValueError("Cannot compute precision for empty datasets.")

    # Total number of labels
    n = y.size

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates the precision of the classifier, or the number of correctly predicted positive labels
        divided by the total number of predicted positive labels

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Precision
    """

    # If predicted labels and true labels are not the same size, throw an error
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size.')
    # If datasets are empty, throw an error
    if len(y) == 0 or len(y_hat) == 0:
        raise ValueError("Cannot compute precision for empty datasets.")
    # If there are no positive cases
    if np.sum(y == 1) == 0:
        raise ValueError("Precision is N/A because there are no positive cases.")

    # Number of false positives (occurrences where predicted is 1 and true is 0 )
    false_positives = np.sum((y == 0) & (y_hat == 1))

    # Number of true positives
    true_positives = np.sum((y == 1) & (y_hat == 1))

    return true_positives / (true_positives + false_positives)


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates the recall of the classifier, or the number of correctly predicted positive labels
        divided by the total number of labels that are actually positive

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Recall, or the true positive rate
    """
    # If predicted labels and true labels are not the same size, throw an error
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size.')
    # If datasets are empty, throw an error
    if len(y) == 0 or len(y_hat) == 0:
        raise ValueError("Cannot compute recall for empty datasets.")
    # If there are no positive cases
    if np.sum(y == 1) == 0:
        raise ValueError("Cannot compute recall because there are no true positive labels.")

    # Number of false negatives (occurrences where predicted is 0 and true is 1)
    false_negatives = np.sum((y == 1) & (y_hat == 0))

    # Number of true positives
    true_positives = np.sum((y == 1) & (y_hat == 1))

    return true_positives / (true_positives + false_negatives)


def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    """
    Calculate ROC curve points as pairs of (false positive rate, true positive rate).

    Args:
        y: True labels.
        p_y_hat: Predicted probabilities of being in labeled as positive.

    Returns:
        A generator of ROC curve points as (FPR, TPR) pairs.

    """

    # If predicted labels and true labels are not the same size, throw an error
    if y.size != p_y_hat.size:
        raise ValueError('y and p_y_hat must be the same shape/size!')
    # If datasets are empty, throw an error
    if len(y) == 0 or len(p_y_hat) == 0:
        raise ValueError("Cannot compute ROC curve for empty datasets.")

    # Sort the probabilities in descending order
    sorted_p_indices = np.argsort(-p_y_hat)
    p_y_hat_sorted = p_y_hat[sorted_p_indices]
    # Sort the labels in the same order
    y_sorted = y[sorted_p_indices]

    # find the number of positive and negative labels
    n_zeros, n_ones = count_label_occurrences(y_sorted)

    # Stores the false positive rate
    false_pos_rate = 0
    # Stores the true positive rate
    true_pos_rate = 0

    # Stores the (FPR, TPR) pairs
    roc_curve = []

    # Iterate through sorted probabilities to calculate true positive rate and false positive rate
    for i in range(len(y_sorted)):
        # If the label is positive, calculate the TP rate
        if y_sorted[i] == 1:
            true_pos_rate += 1 / n_ones
        else:
            # If the label is negative, calculate the FP rate
            false_pos_rate += 1 / n_zeros
        roc_curve.append((false_pos_rate, true_pos_rate))

    return tuple(roc_curve)


def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    """
    Calculates the area under the ROC curve.

    Args:
        y: True labels.
        p_y_hat: Predicted probabilities of being in class 1.

    Returns:
        The area under the ROC curve.

    """

    # If predicted labels and true labels are not the same size, throw an error
    if y.size != p_y_hat.size:
        raise ValueError('y and p_y_hat must be the same shape/size!')
    # If datasets are empty, throw an error
    if len(y) == 0 or len(p_y_hat) == 0:
        raise ValueError("Cannot compute AUC for empty datasets.")

    roc_pairs = roc_curve_pairs(y, p_y_hat)

    # Initialize variables for AUC calculation
    area_under_curve = 0.0
    # Stores the value of the previous false positive rate
    prev_x = 0
    # Stores the value of the previous true positive rate
    prev_y = 0

    # Loop through ROC curve points to calculate AUC
    for x, y in roc_pairs:
        # Use the trapezoidal method for each pair of points (width * (height1+height2)/2)
        area_under_curve += (x - prev_x) * (y + prev_y)/2
        prev_x, prev_y = x, y

    return area_under_curve
