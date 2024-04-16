import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Index of feature to split on
        self.threshold = threshold  # Threshold for the feature
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Predicted value (for leaf nodes)

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stop if max depth is reached or if all samples are of the same class
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        # Find the best split
        best_split = self._best_split(X, y)
        if best_split is None:
            return TreeNode(value=self._most_common_label(y))

        # Recursive splitting
        left_indices = X[:, best_split.feature] <= best_split.threshold
        right_indices = ~left_indices
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(feature=best_split.feature, threshold=best_split.threshold,
                        left=left_subtree, right=right_subtree)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            feature_values = np.unique(X[:, feature])
            for threshold in feature_values:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                gini = self._gini_index(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = Split(feature, threshold)

        return best_split

    def _gini_index(self, left_y, right_y):
        total_samples = len(left_y) + len(right_y)
        gini_left = self._gini_impurity(left_y)
        gini_right = self._gini_impurity(right_y)
        gini_index = (len(left_y) / total_samples) * gini_left + (len(right_y) / total_samples) * gini_right
        return gini_index

    def _gini_impurity(self, y):
        class_counts = Counter(y)
        gini = 1.0
        for class_count in class_counts.values():
            proportion = class_count / len(y)
            gini -= proportion ** 2
        return gini

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_tree(x, self.root) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

class Split:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

#how to use this code:
X_train = np.array([[2, 2], [2, 3], [3, 2], [3, 3], [4, 2], [4, 3]])
y_train = np.array([0, 0, 1, 1, 2, 2])
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X_train, y_train)
predictions = tree.predict(X_train)
print(predictions)
