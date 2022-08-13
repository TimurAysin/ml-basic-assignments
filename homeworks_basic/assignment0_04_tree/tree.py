from queue import Queue
import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    p = np.mean(y, axis=0)
    return -np.sum(p * np.log(p + EPS))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    p = np.mean(y, axis=0)
    return 1 - np.sum(p ** 2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    return np.mean((y - np.mean(y)) ** 2)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index=-1, threshold=-1, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None

class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0        
        self.debug = debug  

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        indices_less = X_subset[:, feature_index] < threshold
        indices_not_less = X_subset[:, feature_index] >= threshold

        X_left = X_subset[indices_less]
        y_left = y_subset[indices_less]

        X_right = X_subset[indices_not_less]
        y_right = y_subset[indices_not_less]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        indices_less = X_subset[:, feature_index] < threshold
        indices_not_less = X_subset[:, feature_index] >= threshold

        y_left = y_subset[indices_less]
        y_right = y_subset[indices_not_less]
        
        return y_left, y_right

    def __calc_criterion(self, y_left, y_right):
        return (y_left.size * self.criterion(y_left) + y_right.size * self.criterion(y_right)) / (y_left.size + y_right.size)

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        
        feature_index = -1
        threshold = -1
        smallest_criterion = np.inf

        for ftr_i in range(X_subset.shape[1]):
            feature_values = list(set((X_subset[:, ftr_i])))
            for thr in feature_values:
                y_left, y_right = self.make_split_only_y(ftr_i, thr, X_subset, y_subset)
                new_criterion = self.__calc_criterion(y_left, y_right)
                if smallest_criterion > new_criterion:
                    smallest_criterion = new_criterion
                    feature_index = ftr_i
                    threshold = thr

        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """        

        if len(y_subset) < self.min_samples_split or self.depth == self.max_depth:
            new_node = Node()

            if self.classification:                
                new_node.value = np.argmax(np.sum(y_subset, axis=0))
                new_node.proba = np.mean(y_subset, axis=0)
            else:
                new_node.value = np.mean(y_subset)

            return new_node

        feature_index, threshold = self.choose_best_split(X_subset, y_subset)

        self.depth += 1

        (X_l, y_left), (X_r, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
        new_node = Node(feature_index, threshold, y_left.size / y_subset.size)
        new_node.left_child = self.make_tree(X_l, y_left)
        new_node.right_child = self.make_tree(X_r, y_right)        

        self.depth -= 1

        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        n_objects = X.shape[0]
        if self.classification:
            y_predictions = np.argmax(self.predict_proba(X), axis=1).reshape(n_objects, 1)
        else:
            y_predictions = np.zeros(n_objects)
            indices = np.arange(n_objects)
            q = Queue()
            q.put((self.root, indices, X))

            while not q.empty():
                node, indices_subset, X_subset = q.get()
                if node.left_child is None:
                    y_predictions[indices_subset] = node.value
                else:
                    (X_left, y_left), (X_right, y_right) = self.make_split(node.feature_index, node.value,
                                                                    X_subset, indices_subset)
                    q.put((node.left_child, y_left, X_left))
                    q.put((node.right_child, y_right, X_right))

        return y_predictions
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        n_objects = X.shape[0]
        y_predicted_probs = np.zeros((n_objects, self.n_classes))
        indices = np.arange(n_objects)

        q = Queue()
        q.put((self.root, indices, X))

        while not q.empty():
            node, indices_subset, X_subset = q.get()
            if node.left_child is None:
                y_predicted_probs[indices_subset] = node.proba
            else:
                (X_left, y_left), (X_right, y_right) = self.make_split(node.feature_index, node.value,
                                                                       X_subset, indices_subset)
                q.put((node.left_child, y_left, X_left))
                q.put((node.right_child, y_right, X_right))

        return y_predicted_probs