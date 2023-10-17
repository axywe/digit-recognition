import numpy as np

accuracy_history = []
loss_history = []


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, load_model=False):
        if load_model:
            self.W1 = np.load('../weights/W1.npy')
            self.b1 = np.load('../weights/b1.npy')
            self.W2 = np.load('../weights/W2.npy')
            self.b2 = np.load('../weights/b2.npy')
        else:
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
            self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        z2_max = np.max(self.z2, axis=1, keepdims=True)
        exp_scores = np.exp(self.z2 - z2_max)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def train(self, X, y, num_passes=10000, epsilon=0.01, batch_size=100):
        for i in range(0, num_passes):
            random_indices = np.random.choice(len(y), size=batch_size, replace=False)
            X_batch = X[random_indices, :]
            y_batch = y[random_indices]

            # Forward pass
            self.forward(X_batch)

            # Backpropagation
            delta3 = self.probs
            delta3[range(len(y_batch)), y_batch] -= 1
            dW2 = np.dot(self.a1.T, delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.W2.T) * (1 - np.power(self.a1, 2))
            dW1 = np.dot(X_batch.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Update weights and biases
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            predicted_labels = np.argmax(self.forward(X_batch), axis=1)
            accuracy = np.mean(predicted_labels == y_batch)
            accuracy_history.append(accuracy)

            loss = -np.log(self.probs[range(len(y_batch)), y_batch])
            total_loss = np.sum(loss)
            loss_history.append(total_loss)
            if i % 100 == 0:
                print(f"Training pass {i} completed.")
        return accuracy_history, loss_history
