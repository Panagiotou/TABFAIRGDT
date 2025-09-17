import numpy as np
import pandas as pd


def discrimination_dataset(y, sensitive):
    """

    :param y: The target values (class labels)
    :param sensitive: The sensitive sample
    :return: The discrimination of dataset
    """
    """
    p0: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 0, clf(𝑥) = +}∣
    p1: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 1, clf(𝑥) = +}∣
    n_zero: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 0}∣
    n_one: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 1}∣
    """
    p0, p1, n_zero, n_one = 0, 0, 0, 0
    for i in range(0, len(y)):
        if sensitive[i] == 0.0:
            n_zero += 1
            if y[i] == 1.0:
                p0 += 1
        elif sensitive[i] == 1.0:
            n_one += 1
            if y[i] == 1.0:
                p1 += 1

    if n_one == 0 and n_zero == 0:
        d = 0
    elif n_zero == 0:
        d = -(p1 / n_one)
    elif n_one == 0:
        d = p0 / n_zero
    else:
        d = (p0 / n_zero) - (p1 / n_one)
    return d


def discrimination(y, y_pred, sensitive):
    """

    :param y: The target values (class labels)
    :param y_pred: The target values (class labels) predicted by the decision tree
    :param sensitive: The sensitive sample
    :return: The discrimination of dataset
    """
    w2, x2, u2, v2, b, b_not = 0, 0, 0, 0, 0, 0
    y_length = len(y)
    for index in range(0, y_length):
        if y_pred[index] == 1:
            if sensitive[index] == 0:
                if y[index] == 0:
                    w2 += 1
                elif y[index] == 1:
                    x2 += 1
            elif sensitive[index] == 1:
                if y[index] == 0:
                    u2 += 1
                elif y[index] == 1:
                    v2 += 1
        if sensitive[index] == 1:
            b += 1
        elif sensitive[index] == 0:
            b_not += 1

    w2 = w2 / y_length
    x2 = x2 / y_length
    u2 = u2 / y_length
    v2 = v2 / y_length

    b = b / y_length
    b_not = b_not / y_length

    return ((w2 + x2) / b_not) - ((u2 + v2) / b)


class Leaf:
    """

    :param path: A list of tuples representing a path to a leaf from the root node
            where a tuple is a leaf in the tree.
            The tuple is in the format (node id, feature, way).
            "Node id" is id of the node in sklearn.
            "Feature" is the feature of a leaf.
            "Way" allows to know if we have to go left or right when we navigate in the tree.
    :type path: tuple
    :param node_id: The id of the node in sklearn.
    :type node_id: int
    :param u: The portion of item of the dataset whose class is negative
            and the sensitive attribute is positive contained by leaf
    :type u: float
    :param v: The portion of item of the dataset whose class is positive
            and the sensitive attribute is positive contained by leaf
    :type v: float
    :param w: The portion of item of the dataset whose class is negative
            and the sensitive attribute is negative contained by leaf
    :type w: float
    :param x: The portion of item of the dataset whose class is positive
            and the sensitive attribute is negative contained by leaf
    :type x: float
    :param transactions: A list of sample indexes used by the leaf
    :type transactions: list
    """

    def __init__(self, node_id, u, v, w, x, transactions=None):
        self.node_id = node_id
        self.acc = None
        self.disc = None
        self.ratio = None
        self.u = u
        self.v = v
        self.w = w
        self.x = x
        self.transactions = transactions

    def accuracy(self, cnt_p, cnt_n, portion_zero, portion_one):
        n = self.u + self.w
        p = self.v + self.x
        """"
        WARNING ! Don't use '(self.u + self.w) > (self.v + self.x)' or 'p>n'
        self.u, self.w,... are fractions, so in some cases this is not precise and causes a bug.
        (can be caused by python rounding during a division)
        cnt_p and cnt_n are the number of positive and negative class,
        thus integers, there will be no error when using them.
        """
        if cnt_p > cnt_n:
            self.acc = n - p
            self.disc = (self.u + self.v) / portion_one - (self.w + self.x) / portion_zero

        else:
            self.acc = p - n
            self.disc = -(self.u + self.v) / portion_one + (self.w + self.x) / portion_zero

        if self.acc == 0:
            """
            In theory, if the accuracy is 0 and the discrimination
            after relabeling (self.disc) is < 0, this leaf must be one of
            the best to relabeling because we will have a loss in discrimination
            but no loss in accuracy.
            This is why a positive value very close to 0 is used to avoid a division by 0
            and to maintain a high ratio.
            """
            self.ratio = self.disc / -0.00000000000000000000000000000000000001
        else:
            self.ratio = self.disc / self.acc

    def __str__(self):
        return f"Path: format -> (node id, feature, way)\n{self.path} " \
               f"\nnode_id: {self.node_id} " \
               f"\nThe effect of relabeling the leaf on accuracy: {self.acc}" \
               f"\nThe effect of relabeling the leaf on discrimination: {self.disc} " \
               f"\nratio: {self.ratio} " \
               f"\ncontingency table: \n{[self.u, self.v]}\n{[self.w, self.x]}" \
               f"\ntransactions: {self.transactions}"

    def __repr__(self):
        return f"{self.node_id}"


def get_transactions_by_leaf(clf, path, x):
    """
    Allows to retrieve the indexes of the samples of a leaf.

    :param clf: The decision tree.
    :param path: A list of tuples representing a path to a leaf from the root node
            where a tuple is a leaf in the tree.
            The tuple is in the format (node id, feature, way).
            "Node id" is id of the node in sklearn.
            "Feature" is the feature of a leaf.
            "Way" allows to know if we have to go left or right when we navigate in the tree.
    :param x: The training input samples.
    :return: A list of sample indexes used by the leaf.
    """
    filtered = pd.DataFrame(x)
    for tupl in path:
        node_id = tupl[0]
        feature = tupl[1]
        if tupl[2] == 'left':
            filtered = filtered.loc[filtered[feature] <= clf.tree_.threshold[node_id]]
        elif tupl[2] == 'right':
            filtered = filtered.loc[filtered[feature] > clf.tree_.threshold[node_id]]
        else:
            raise Exception("Should not reach here")
    return list(filtered.index)


def get_leaves_candidates(clf, X, y, sensitive):
    """
    Recover leaves that could be used for relabeling, without recursion.

    :param clf: The decision tree classifier.
    :param X: The training input samples.
    :param y: The target values (class labels).
    :param sensitive: The sensitive attribute values.
    :param cnt: Tuple where the index 0 is the count of negative sensitive classes,
                and index 1 is the count of positive sensitive classes.
    :param length: The total number of samples.
    :return: A list of `Leaf` objects representing candidate leaves for relabeling.
    """
    # Get leaf indices for all samples
    leaf_indices = clf.apply(X)
    cnt = np.unique(sensitive, return_counts=True)[1]

    length = len(y)
    leaves = []
    
    # Group samples by leaf
    unique_leaves = np.unique(leaf_indices)
    for leaf_index in unique_leaves:
        # Get the samples in this leaf
        samples_in_leaf = np.where(leaf_indices == leaf_index)[0]
        
        # Compute fractions for this leaf
        u = sum((sensitive[samples_in_leaf] == 1) & (y[samples_in_leaf] == 0)) / length
        v = sum((sensitive[samples_in_leaf] == 1) & (y[samples_in_leaf] == 1)) / length
        w = sum((sensitive[samples_in_leaf] == 0) & (y[samples_in_leaf] == 0)) / length
        x = sum((sensitive[samples_in_leaf] == 0) & (y[samples_in_leaf] == 1)) / length

        

        # Create Leaf object
        leaf = Leaf(node_id=leaf_index, u=u, v=v, w=w, x=x, transactions=samples_in_leaf)
        
        # Calculate accuracy and discrimination impact
        leaf.accuracy(v + x, u + w, cnt[0] / length, cnt[1] / length)

        
        # Add leaf to candidates if discrimination impact is negative
        if leaf.disc < 0:
            leaves.append(leaf)
            # if leaf.acc > 0:
            #     print("Acc impact", leaf.acc)
            #     print("Disc impact", leaf.disc)

    
    return leaves

def acc_loss(leaves):
    """
    Calculate the remaining accuracy after relabeling specified leaves.
    
    Args:
        acc_initial: Initial accuracy of the tree
        leaves: The leaves that we will keep to relabel
    Returns:
        The new accuracy we will get
    """
    acc_loss = 0
    for leaf in leaves:
        # print(leaf.acc)
        # if leaf.acc < -acc_threshold:
        acc_loss += leaf.acc
    return acc_loss

# rem disc(𝐿) := disc 𝑇 + ∑ Δdisc 𝑙 ≤ 𝜖
def rem_disc(disc_tree, leaves, threshold):
    """

    :param disc_tree: The discrimination of the tree.
    :param leaves: The leaves that we will keep to relabel them.
    :param threshold: The threshold of discrimination that we do not want to exceed.
    :return: The new discrimination we will get.
    """
    s = 0
    for leaf in leaves:
        if leaf.disc < threshold:
            s += leaf.disc
    return disc_tree + s


def leaves_to_relabel(clf, x, y, y_pred, sensitive, acc_threshold=-1, disc_threshold=0):
    """
    Select exactly this set of leaves that is "optimal" w.r.t. reducing the discrimination with
    minimal loss in accuracy.

    :param clf: The decision tree.
    :param x: The training input samples.
    :param y: The target values (class labels).
    :param y_pred: The target values (class labels) predicted by the decision tree.
    :param sensitive: The sensitive sample.
    :param disc_threshold: The threshold of discrimination that we do not want to exceed.
    :return: The leaves that we will keep to relabel them.
    :acc_threshold: Max percentage drop in accuracy.
    """
    disc_tree = discrimination(y, y_pred, sensitive)

    if disc_tree < 0:
        sensitive = 1 - sensitive  # Flip encoding if needed
        disc_tree = discrimination(y, y_pred, sensitive)


    initial_acc = np.mean(y == y_pred)

    # ℐ := { 𝑙 ∈ ℒ ∣ Δdisc 𝑙 < 0 }
    i = get_leaves_candidates(clf, x, y, sensitive)

    # 𝐿 := {}
    leaves = set()
    # while rem disc(𝐿) > 𝜖 do

    # print((initial_acc + acc_loss(leaves)) / initial_acc )
    # while rem_disc(disc_tree, leaves, threshold) > threshold and i:
    while( 
        rem_disc(disc_tree, leaves, disc_threshold) > disc_threshold 
        and (acc_threshold <= 0 or 1 - (initial_acc + acc_loss(leaves)) / initial_acc <= acc_threshold) 
        and i
        ):

        # rem_acc(initial_acc, leaves, acc_threshold)
        # best l := arg max 𝑙∈ℐ∖𝐿 (disc 𝑙 /acc 𝑙 )
        best_l = i[0]
        for leaf in i:
            if leaf.ratio > best_l.ratio:
                best_l = leaf
        # 𝐿 := 𝐿 ∪ {𝑙}
        leaves.add(best_l)
        i.remove(best_l)

    print("Init acc", initial_acc, "New acc", initial_acc + acc_loss(leaves))

    if rem_disc(disc_tree, leaves, disc_threshold) > disc_threshold:
        print("\033[1;33m" + "Unable to reach the threshold." + "\033[0m")

    return leaves


def browse_and_relab(clf, node_id):
    """
    Relabel one leafs of the decision tree.

    :param clf: The decision tree.
    :param node_id: The id of the node (leaf) to be relabeled.
    """

    if clf.tree_.value[node_id][0][0] == clf.tree_.value[node_id][0][1]:
        clf.tree_.value[node_id][0][1] += 1
    else:
        clf.tree_.value[node_id][0][0], clf.tree_.value[node_id][0][1] = \
            clf.tree_.value[node_id][0][1], clf.tree_.value[node_id][0][0]


def relabeling(clf, x, y, y_pred, sensitive, disc_threshold=0):
    """
    Relabel the leaves of the decision tree so that the discrimination decreases,
    until it falls below the threshold, while minimizing the loss of accuracy.

    :param clf: The decision tree.
    :param x: The training input samples.
    :param y: The target values (class labels).
    :param y_pred: The target values (class labels) predicted by the decision tree.
    :param sensitive: The sensitive sample.
    :param threshold: The threshold of discrimination that we do not want to exceed.
    :return:
    """
    if discrimination_dataset(y, sensitive) < 0:
        print(discrimination_dataset(y, sensitive))
        raise Exception("The discrimination of the dataset can't be negative.")
    if len(np.unique(sensitive)) != 2:
        raise Exception("Only two different labels are expected for the sensitive sample.")
    if len(np.unique(y)) != 2:
        raise Exception("Only two different labels are expected for the class sample.")

    for leaf in leaves_to_relabel(clf, x, y, y_pred, sensitive, disc_threshold=disc_threshold):
        if clf.tree_.value[leaf.node_id][0][0] == clf.tree_.value[leaf.node_id][0][1]:
            clf.tree_.value[leaf.node_id][0][1] += 1
        else:
            clf.tree_.value[leaf.node_id][0][0], clf.tree_.value[leaf.node_id][0][1] = \
                clf.tree_.value[leaf.node_id][0][1], clf.tree_.value[leaf.node_id][0][0]