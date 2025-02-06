import torch as t


def from_pytorch():
    input = t.tensor(1)
    expected = t.tensor(2)
    weight = t.tensor(1.0, requires_grad=True)

    prediction = weight * input
    loss = (prediction - expected) ** 2

    print(loss)

    loss.backward()
    print(weight.grad)


def from_scratch():
    value = 1
    expected = 2
    weight = 1.0

    prediction = weight * value
    loss = (prediction - expected) ** 2

    print(loss)

    # Compute the gradient
    backward_pass_grad = 2 * (prediction - expected) * value

    print(backward_pass_grad)


from_pytorch()
from_scratch()
