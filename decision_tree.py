import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

DEBUG = False

class Node:
    def __init__(self, dimension=None, feature=None, thresh=None, error=None, parent_node=None, dataset=None, data_labels=None, is_root=False, most_freq=None):
        # updated parameters before split
        self.dimension = dimension
        self.feature = feature
        self.thresh = thresh
        self.error = error
        self.left_node = None
        self.right_node = None
        self.parent_node = parent_node
        self.root = is_root
        self.dataset = dataset
        self.data_labels = data_labels
        self.most_freq_label = most_freq

    def update_variables(self, m_dimension, m_thresh, m_feature, m_error):
        '''
        Updates private member variables, 'm' stands for member
        '''
        self.dimension = m_dimension
        self.feature = m_feature
        self.thresh = m_thresh
        self.error = m_error
        return
    
    def update_leaf_nodes(self, m_left_node, m_right_node):
        '''
        Updates left and right child nodes
        -- only happens when this node contains the optimal split
        '''
        self.left_node = m_left_node
        self.right_node = m_right_node
        return
    
    def update_parent(self, parent):
        self.parent_node = parent
    
    def get_variables(self):
        return self.dimension, self.feature, self.thresh, self.most_freq_label
    
    def get_error(self):
        return self.error
    
    def __get_dataset(self):
        return self.dataset
    
    def get_dataset(self):
        return self.dataset.copy()
    
    def __get_labels(self):
        return self.data_labels


class DecisionTree:

    def __init__(self):
        self.tree = None
        self.leaf_nodes = 1
        self.current_nodes: list[Node] = []
        self.queued_nodes: list[Node] = []


    def find_split(self, features, labels, node):
        '''
        Takes in a node along with a dataset and finds the best split based on 0-1 loss.
        The optimal split data is stored in the node for future comparison with other nodes.

        Parameters:
        -- features: (45, x) vector containing the results of the x values on the 45 features.
        -- labels: (x,) vector containing the labels of the x values.
        -- node: current node to be evaluated
        '''

        # Step 1: Iterate through each feature and test thresholds, 0-1 loss
        # Step 2: Store the best feature and threshold as your iterate
        
        # instantiate variables
        itr = 0
        thresh = 0
        optimal_dimension = 0
        optimal_thresh = 0
        optimal_error = 0
        optimal_feature = [0, 1]

        for i in range(10):
            for j in range(i + 1, 10):
                if (len(features[itr]) == 0):
                    continue
                
                # threshold depends on the spread of the data
                thresh_spread = 1
                feature_size = len(features[itr])
                ''' --- TEST: Adjusting step size ---
                # if feature_size < 3000 and feature_size > 600:
                #     thresh_spread = 3
                # elif  feature_size < 500:
                #     thresh_spread = 5
                '''
                low_thresh = np.min(features[itr])
                high_thresh = np.max(features[itr])
                interval = abs(high_thresh - low_thresh) / (60 * thresh_spread)

                # Indicates that the feature cannot differentiate between the classes
                if low_thresh == high_thresh:
                    itr +=1
                    continue

                # Use the mean when the list becomes small
                if (feature_size < 1350):
                    thresh = np.mean(features[itr])
                    predictions = np.where(features[itr] > thresh, j, i)
                    zero_one_loss = np.sum(predictions == labels)

                    # no need to split if the parent error is the same
                    if (node.parent_node is not None and zero_one_loss == node.parent_node.get_error()):
                            continue
                    
                    # condition for splitting a node with only 2 labels
                    if (zero_one_loss == 2):
                        thresh = features[itr][0]
                        zero_one_loss = 1

                    # store the parameters if the error is improved, prohibit the previous feature
                    if ((node.parent_node is None or (node.parent_node.most_freq_label != j and node.parent_node.most_freq_label != i)) \
                        and zero_one_loss != len(labels) and zero_one_loss > optimal_error):
                        optimal_dimension = itr
                        optimal_feature = [i, j]
                        optimal_thresh = thresh
                        optimal_error = zero_one_loss

                else:
                    thresh = low_thresh
                    while (thresh < high_thresh):
                        # Test threshold accuracy using 0-1 Loss
                        predictions = np.where(features[itr] > thresh, j, i) # if true j, else i
                        zero_one_loss = np.sum(predictions == labels)

                        # no need to split if the parent error is the same
                        if (node.parent_node is not None and zero_one_loss == node.parent_node.get_error()):
                            thresh += interval
                            continue

                        # store the parameters if the error is improved, prohibit the previous feature
                        if ((node.parent_node is None or (node.parent_node.most_freq_label != j and node.parent_node.most_freq_label != i)) \
                            and zero_one_loss != len(labels) and zero_one_loss > optimal_error):
                            optimal_dimension = itr
                            optimal_feature = [i, j]
                            optimal_thresh = thresh
                            optimal_error = zero_one_loss

                        thresh += interval
                itr += 1

        # store optimal split data in node
        node.update_variables(optimal_dimension, optimal_thresh, optimal_feature, optimal_error)

        ''' ----- TEST PRINTS -----
        # test plot
        # plt.figure()
        # plt.scatter(features[optimal_dimension][:1000], labels[:1000])
        # plt.show()

        # print()
        # print("optimal_dimensions:", optimal_dimension)
        # print("optimal_feature:", optimal_feature)
        # print("optimal_thresh:", optimal_thresh)
        # print("optimal_error:", optimal_error)
        # predictions = np.where(node.dataset[optimal_dimension] > optimal_thresh,  optimal_feature[1], optimal_feature[0])
        # ----- TEST PRINTS ----- '''

        return

    def train(self, features, labels, num_leaves):
        '''
        Train a decision tree on the given features and labels.
        
        Parameters:
        -- features: (45, x) vector containing the results of the x values on the 45 features.
        -- labels: (x,) vector containing the labels of the x values.
        -- num_leaves: The number of leaves in the decision tree.
        '''

        # we start with the root
        self.tree = Node(is_root = True, dataset=features, data_labels=labels)
        self.current_nodes.append(self.tree)

        prev_dimension = [-1] # prevents the child from splitting on the same criteria as the parent
        while num_leaves > self.leaf_nodes:
            # Step 1. This function is going to come up with some possible candidates for a split
            # Step 2. we have to evaluate the results of each one, and choose which node to split
            # Step 3. We must also store the information necessary so that we don't have to rerun the algorithm on
            # certain nodes, we just have it ready
            best_current_node = self.current_nodes[0]

            # check current nodes to see which one has the best 0-1 error
            for node in self.current_nodes:
                if (node.get_error() == None):
                    self.find_split(node._Node__get_dataset(), node._Node__get_labels(), node) # this updates the error in node

                # print("current node stats:", node.get_variables(), node.get_error())
                if node.get_error() > best_current_node.get_error() and node.dimension not in prev_dimension:
                    best_current_node = node

            if DEBUG:
                print(f"CHOSEN SPLIT: {best_current_node.get_variables()}, 0-1 LOSS: {best_current_node.get_error()}")
            
            # Load best node information
            dimension, _, thresh, _ = best_current_node.get_variables()
            dataset = best_current_node._Node__get_dataset()
            labels = best_current_node._Node__get_labels()
            prev_dimension[0] = dimension
            
            # split data and store most freq. value
            mask = dataset[dimension] > thresh
            left_split = dataset[:, mask] # the accurately classified ones
            right_split = dataset[:, ~mask]
            left_labels = labels[mask]
            right_labels = labels[~mask]
            left_most_freq = stats.mode(left_labels)[0]
            right_most_freq = stats.mode(right_labels)[0]

            # store information in the current node
            left_node = Node(dataset=left_split, data_labels=left_labels, most_freq=left_most_freq)
            right_node = Node(dataset=right_split, data_labels=right_labels, most_freq=right_most_freq)
            best_current_node.update_leaf_nodes(left_node, right_node)
            left_node.update_parent(best_current_node)
            right_node.update_parent(best_current_node)
            self.current_nodes.append(left_node)
            self.current_nodes.append(right_node)
            self.leaf_nodes += 1

            # update amount of leaf nodes and remove the node we just split from
            self.current_nodes.remove(best_current_node)    

        return
        
    def predict(self, x):
        '''
        Predict the label for a given input x by traversing the decision tree.
        Assumes that each x is a feature vector

        Parameters:
        -- x: size (45,), The input vector for prediction

        Returns:
        -- label: The predicted label (integer value corresponding to class).
        '''
        current_node = self.tree  # Start at root

        # Traverse the tree until we reach a leaf node
        while current_node.left_node is not None and current_node.right_node is not None:
            dimension, _, thresh, most_freq = current_node.get_variables()

            # Compare the value of x at the optimal dimension (feature index) with the threshold
            if x[dimension] > thresh:
                current_node = current_node.left_node
            else:
                current_node = current_node.right_node

        dimension, _, thresh, most_freq = current_node.get_variables()
        return most_freq
