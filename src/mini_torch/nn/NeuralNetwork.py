import numpy as np

from .Layers import Layer, DenseLayer
from .ActivationFunctions import ReLU, Softmax
from .LossFunctions import LossFunction, MeanSquaredError

LEARNING_RATE = 0.01


class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss_function: LossFunction):
        self.layers = layers
        self.loss_function = loss_function

    def forward_prop(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.loss_function.calculate_loss(y_true, y_pred)

    def backward_prop(self, loss: float):
        pass

    def train(self, x: np.ndarray, y_true: np.ndarray, learning_rate: float, epochs: int):
        for epoch in range(epochs):
            y_pred = self.forward_prop(x)
            loss = self.loss_function.calculate_loss(y_true, y_pred)
            self.backward_prop(loss)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward_prop(x).T[0]


def main():
    nn = NeuralNetwork(
        [
            DenseLayer(input_size=64, output_size=32, activation_function=ReLU()),
            DenseLayer(input_size=32, output_size=16, activation_function=ReLU()),
            DenseLayer(input_size=16, output_size=8, activation_function=ReLU()),
            DenseLayer(input_size=8, output_size=3, activation_function=Softmax()),
        ],
        loss_function=MeanSquaredError(),
    )

    print(nn.layers[0].weights.shape)
    print(nn.layers[1].weights.shape)

    x = np.random.randn(64, 30)  # 30 examples, 64 features each
    print(x.shape)
    output = nn.forward_prop(x).T
    print(output)

    x = np.random.randn(
        64, 1
    )  # 1 example (what we want to predict based on our features), 64 features
    output = nn.predict(x)
    print(f"\n\nPrediction:\n{output}")


if __name__ == "__main__":
    main()


