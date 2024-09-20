from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()


def train(all_images, all_labels, label1, label2):
    # print("Training on labels: ", label1, label2)

    # loop through the labels and only store the images that are 
    # labeled 1 or 2
    images = []
    labels = []

    for i in range(len(all_labels)):
        if all_labels[i] == label1 or all_labels[i] == label2:
            images.append([1] + all_images[i])
            labels.append(all_labels[i])

    images = np.array(images, dtype=np.float32).T # (785, x)
    labels = np.array(labels, dtype=np.float32)   # (x,)
    assert images.shape[0] == 785, "images shape is wrong"

    Xy = images.dot(labels)
    XX_t = images.dot(images.T) + + 1e-5 * np.eye(785)
    XX_t = np.linalg.inv(XX_t)


    # get weights
    w = XX_t.dot(Xy)

    return w


# required for graduate students only
def get_optimal_thresh(images_train, labels_train, label1, label2, w):
    pass


def test(all_images_test, all_labels_test, label1, label2, w, thresh):
    # We will test if our predictions are working by multiplying our 
    # weights by the test data and classifying based on the threshold

    images = []
    labels = []

    for i in range(len(all_labels_test)):
        if all_labels_test[i] == label1 or all_labels_test[i] == label2:
            images.append([1] + all_images_test[i])
            labels.append(all_labels_test[i])
    
    # test
    predictions = np.zeros(len(labels))
    for i in range(len(labels)):
        predictions[i] = np.dot(w, images[i])

    # print(predictions[:100])
    # predictions = np.where(predictions > np.sum(predictions) / len(predictions), label2, label1)
    predictions = np.where(predictions > thresh, label2, label1)

    print(np.sum(predictions == labels) / len(labels), f"({label1}, {label2})")


if __name__ == "__main__":

    #=======================================================#
    #                       Instructions                    #
    # pip install                                           #
    # - mnist                                               #
    # - numpy                                               #
    # - matplotlib                                          #
    #                                                       #
    # Afterwards run the python file like normal            #
    # - Accuracy will be displayed for each combination for # 
    #   10 choose 2 total                                   #
    #=======================================================#

    # load the data
    mndata = MNIST('./datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    # train regression model on all combinations of 2 labels given the 10 total labels
    itr = 0
    for i in range(10):
        for j in range(i+1, 10):
            w = train(images_list, labels_list, i, j)
            assert w.shape == (785,), "weights shape is wrong"

            # test the model
            if (i == 5 and j == 8):
                test(images_list_test, labels_list_test, i, j, w, w[0] * 1.25)
            else:
                test(images_list_test, labels_list_test, i, j, w, w[0])

            itr += 1

    print("iterations:", itr)


    
