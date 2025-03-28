from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from decision_tree import DecisionTree
import time


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()

def load_weights(file_name, num_lines, dim):
    w = np.zeros((num_lines, dim))
    b = np.zeros((num_lines, 1))

    count = 0

    with open(file_name) as file:
        for line in file:
            values = np.array([float(i) for i in line.split(',')])

            w[count, :] = values[0:dim]
            b[count] = values[dim]

            count += 1

    return (w, b)

def test(w_X, labels_train, w_X_test, images_list_test, labels_list_test, num_leaves, extensive_test=True):
    if extensive_test == True:
        test_leaf_values = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        accuracy = [0]  * len(test_leaf_values)

        for i in range(len(test_leaf_values)):
            decision_tree = DecisionTree()
            decision_tree.train(w_X, labels_train, test_leaf_values[i])
            results = []
            for j in range(len(images_list_test)):
                predicted_label = decision_tree.predict(w_X_test[:, j])  # w_X[:, i] is the feature vector for sample i
                results.append(predicted_label)

            # store accuracy
            accuracy[i] = sum(results == labels_list_test) / len(labels_list_test)
            print(f"Progress: {round(100 * (i+1)/len(accuracy), 2)}%; Accuracy {round(accuracy[i] * 100, 4)}%")


        # Plot the accuracy vs the # of leaves
        plt.figure(figsize=(10, 6))
        plt.plot(test_leaf_values, accuracy, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel("Number of Leaf Nodes", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Decision Tree Accuracy vs Number of Leaf Nodes", fontsize=14)
        plt.grid(True)
        plt.show()

        
    else:
        # train decision tree and evaluate on specified dataset
        decision_tree = DecisionTree()
        decision_tree.train(w_X, labels_train, num_leaves)

        results = []
        for i in range(len(images_list_test)):
            predicted_label = decision_tree.predict(w_X_test[:, i])  # w_X[:, i] is the feature vector for sample i
            results.append(predicted_label)

        results = np.array(results)
        print(f"Accuracy: {round(sum(results == labels_list_test) / len(labels_list_test), 4) * 100}%")

if __name__ == "__main__":
    '''
    Instructions:
    - Step 1: Install necessary libraries
    - Step 2: Specify extensive_test as True for multiple leaf evaluation and plotting
        - False if you want to test a specific number of leaves, specify num_leaves = #
    - Step 3: Run decision_tree_classification.py file
    '''
    
    ''' Load the dataset '''
    mndata = MNIST('./datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    # Use this only if you need to use the provided weights
    num_lines = 45
    (w, b) = load_weights('weights.txt', num_lines, 28 * 28)

    images_train = (np.asarray(images_list).astype(float)) #  shape: (60000, 785)
    labels_train = np.asarray(labels_list).astype(float) #  shape: (60000,)
    images_list_test = (np.asarray(images_list_test).astype(float))
    labels_list_test = np.asarray(labels_list_test).astype(float)

    ''' Get Features '''
    # 1. merge bias and weights
    # 2. add 1 to the 0th entry of images
    # 3. evaluate w times the inputs, obtaining 45 features for every x
    w = np.concatenate((b, w), axis=1)
    ones = np.ones((len(images_train), 1))
    ones_test = np.ones((len(images_list_test), 1))
    images_train = np.concatenate((ones, images_train), axis=1)
    images_list_test = np.concatenate((ones_test, images_list_test), axis=1)
    w_X = np.dot(w, images_train.T) # expected (45, 60000)
    w_X_test = np.dot(w, images_list_test.T)

    # Standardize
    mean_w_X = np.mean(w_X, axis=1, keepdims=True)
    std_w_X = np.std(w_X, axis=1, keepdims=True)
    w_X = (w_X - mean_w_X) / std_w_X

    w_X_test = (w_X_test - mean_w_X) / std_w_X


    ''' Decision Tree: TRAIN AND TEST ACCURACY EVALUATION '''
    num_leaves = 1700

    # Train Accuracy
    print("Testing on train dataset")
    start_time = time.time()
    ########################################################
    # Set extensive test to True for test from [10 - 2000] #
    # Takes approx. 474.322 seconds on my system           #
    ########################################################
    test(w_X, labels_train, w_X, images_train, labels_train, num_leaves, extensive_test=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds\n")

    # Test Accuracy
    print("Testing on test dataset")
    start_time = time.time()
    test(w_X, labels_train, w_X_test, images_list_test, labels_list_test, num_leaves, extensive_test=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
