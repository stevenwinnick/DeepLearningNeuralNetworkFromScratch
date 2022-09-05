"""
A script to test StevensNeuralNet using a batch of identical data vectors, which cause the neural network's loss to
continuously decrease, as expected.
"""

import StevensNeuralNet


def main():

    # 7 input parameters, 3 hidden layers of size 20, 10 possible output classifications
    neural_net = StevensNeuralNet.MyNeuralNet([7, 20, 20, 20, 10])

    #       /---------inputs---------\  /------classifications------\
    data = [[.8, .6, .7, .5, .3, 0, .9, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [.8, .6, .7, .5, .3, 0, .9, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [.8, .6, .7, .5, .3, 0, .9, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [.8, .6, .7, .5, .3, 0, .9, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [.8, .6, .7, .5, .3, 0, .9, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]

    """
    Training on this data set, we hope to see the loss continuously decrease, as the network learns to classify this 
    specific input vector.
    """
    for i in range(1000):
        neural_net.train(data)
        print("Loss: ", neural_net.current_loss)


if __name__ == "__main__":
    main()
