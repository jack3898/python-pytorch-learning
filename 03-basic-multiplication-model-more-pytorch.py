import torch as t


# Continuing on from 02, this does the same thing, but it uses even more built-in pytorch utilities.
# This is a full-pytorch implementation! Before, it was using some custom logic.
def from_pytorch():
    input_tensor = t.tensor([[1], [2], [3], [4]], dtype=t.float32)
    expected_tensor = t.tensor([[2], [4], [6], [8]], dtype=t.float32)

    test_dataset = t.tensor([5], dtype=t.float32)

    n_samples, n_features = input_tensor.shape

    print(n_samples, n_features)

    model = t.nn.Linear(n_features, n_features)

    initial_prediction = model(test_dataset).item()

    epochs = 1000
    learning_rate = 0.01

    loss = t.nn.MSELoss()  # Mean squared error
    optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic gradient descent

    for epoch in range(epochs):
        prediction = model(input_tensor)
        loss_val = loss(expected_tensor, prediction)

        loss_val.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0:
            [w, _] = model.parameters()
            print(f"Epoch: {epoch}, Loss: {loss_val}, Weight: {w[0][0].item()}")

    print(f"Initial prediction: {initial_prediction:.3f}")
    print(f"Final prediction: {model(test_dataset).item():.3f}")
    print(f"Final weight: {model.weight.item():.3f}")


if __name__ == "__main__":
    from_pytorch()
