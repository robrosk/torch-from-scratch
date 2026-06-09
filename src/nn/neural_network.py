import numpy as np

from src.utilities import Tensor

from .modules import Layer, DenseLayer, ReLU, Softmax, LossFunction, MeanSquaredError

LEARNING_RATE = 0.01


class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss_function: LossFunction):
        self.layers = layers
        self.loss_function = loss_function

    def forward_prop(self, x: Tensor | np.ndarray) -> Tensor:
        x = x if isinstance(x, Tensor) else Tensor(x)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def calculate_loss(self, y_true: Tensor | np.ndarray, y_pred: Tensor) -> Tensor:
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        return self.loss_function.calculate_loss(y_true, y_pred)

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def backward_prop(self, loss: Tensor):
        pass

    def train(self, x: Tensor | np.ndarray, y_true: Tensor | np.ndarray, learning_rate: float, epochs: int):
        for epoch in range(epochs):
            self.zero_grad()
            y_pred = self.forward_prop(x)
            loss = self.calculate_loss(y_true, y_pred)
            self.backward_prop(loss)
            # TODO(you): once backward_prop fills .grad, step the parameters:
            # for layer in self.layers:
            #     layer.update(learning_rate)

    def predict(self, x: Tensor | np.ndarray) -> np.ndarray:
        # inference exits the graph — return raw numpy
        return self.forward_prop(x).data.T[0]


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
    output = nn.forward_prop(x).data.T
    print(output)

    x = np.random.randn(
        64, 1
    )  # 1 example (what we want to predict based on our features), 64 features
    output = nn.predict(x)
    print(f"\n\nPrediction:\n{output}")


if __name__ == "__main__":
    main()
