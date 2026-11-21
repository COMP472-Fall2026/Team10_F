import numpy as np


def gini(y, n_classes):
    """Returns: float, Gini impurity of group y"""
    return 1 - (np.sum((np.bincount(y, minlength=n_classes) / len(y)) ** 2))


def split_score(y_left, y_right, n_classes):
    """Returns: float, Gini average on each side, weighted by their size"""
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * gini(y_left, n_classes) + (len(y_right) / n) * gini(
        y_right, n_classes
    )


def best_split(X, y, n_classes, n_thresholds=10, feat_indices=None):
    """X : (n, d), y : (n,)
    feat_indices: features to try (None = all of them; the Random Forest passes a random subset)
    Returns: (best_feature, best_threshold, best_score)
             the question "X[:, best_feature] <= best_threshold" with the lowest score,
             or (None, None, inf) if no feature can split the group"""
    best_feature = None
    best_threshold = None
    best_score = float("inf")
    if feat_indices is None:  # Normal tree: every feature is tried
        feat_indices = range(X.shape[1])
    for i in feat_indices:
        unique_seuils = np.unique(X[:, i])[:-1]
        if len(unique_seuils) == 0:
            continue
        if len(unique_seuils) <= n_thresholds:
            seuils = unique_seuils
        else:
            seuils = np.quantile(unique_seuils, np.linspace(0,1,n_thresholds+2)[1:-1])
        for t in seuils:
            mask = X[:, i] <= t
            yes = y[mask]
            no = y[~mask]
            score = split_score(yes, no, n_classes)
            if score < best_score:
                best_feature=i
                best_threshold = t
                best_score = score
    return (best_feature, best_threshold, best_score)

def make_leaf(y, n_classes):
    """Returns: dict {"leaf": True, "class": majority class of y}"""
    return {"leaf": True, "class": np.argmax(np.bincount(y))}

def build_tree(X, y, n_classes, depth=0, max_depth=10, min_samples_split=2, n_thresholds=10,
               n_features=None, rng=None):
    """n_features: None = every split tries all the features (normal Decision Tree)
                   int k = every split tries k random features, drawn with rng (used by the Random Forest)
    Returns: dict, the tree (a leaf or a question node)"""
    if depth == max_depth or len(y) < min_samples_split or max(np.bincount(y)) == len(y):
        return make_leaf(y, n_classes)
    feat_indices = None
    if n_features is not None:  # Random Forest: k random features for this split
        feat_indices = rng.choice(X.shape[1], size=n_features, replace=False)
    feature, threshold, _  = best_split(X, y, n_classes, n_thresholds, feat_indices)
    if feature is None:
        return make_leaf(y, n_classes)
    mask = X[:, feature] <= threshold
    left = build_tree(X[mask], y[mask], n_classes, depth + 1, max_depth, min_samples_split, n_thresholds,
                      n_features, rng)
    right = build_tree(X[~mask], y[~mask], n_classes, depth + 1, max_depth, min_samples_split, n_thresholds,
                       n_features, rng)
    return {"leaf": False, "feature": feature, "threshold": threshold, "left": left, "right": right}



def predict_one(tree, x):
    """x : (d,) one point
    Returns: the predicted class for x"""
    if tree["leaf"] == True:
        return tree["class"]
    else:
        if x[tree["feature"]] <= tree["threshold"]:
            return predict_one(tree["left"], x)
        else:
            return predict_one(tree["right"], x)

def predict(tree, X):
    """X : (m, d)
    Returns: array (m,) of predicted classes"""
    predicted = []
    for x in X:
        leaf = predict_one(tree, x)
        predicted.append(leaf)
    return np.array(predicted)
