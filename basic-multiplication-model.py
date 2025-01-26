import numpy as np

# This is a simple model that multiplies the input by a weight
# and tries to predict the expected output. As each epoch passes, 
# the model adjusts the weight to minimize the loss until hopefully
# the model converges to the expected output.
# This is a simple example of a linear regression reference model without use of deep learning libraries, just numpy.
def from_scratch():
    input_arr = np.array([1, 2, 3, 4], dtype=np.float32)
    expected_arr = np.array([2, 4, 6, 8], dtype=np.float32)

    weight = 0.0

    def forward(input):
        return weight * input
    
    def loss(expected, prediction):
        return ((prediction-expected) ** 2).mean()

    def gradient(input, expected, prediction):
        return np.dot(2 * input, prediction - expected).mean()
    
    print("Initial weight: ", weight)
    print("Initial prediction: ", forward(input_arr))

    epochs = 20
    learning_rate = 0.01

    for epoch in range(epochs):
        prediction = forward(input_arr)
        loss_val = loss(expected_arr, prediction)
        grad_val = gradient(input_arr, expected_arr, prediction)

        weight -= learning_rate * grad_val

        if epoch % 2 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_val}, Weight: {weight}")
 
    print("Final weight: ", weight)
    print("Final prediction: ", forward(input_arr))

from_scratch()