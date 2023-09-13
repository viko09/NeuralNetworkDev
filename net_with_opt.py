import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        # ... (código de inicialización existente)

        # Variables auxiliares para Adam
        self.m = [np.zeros(b.shape) for b in self.biases]
        self.v = [np.zeros(b.shape) for b in self.biases]
        self.beta1 = 0.9  # Hiperparámetro de momento
        self.beta2 = 0.999  # Hiperparámetro de la segunda media móvil
        self.epsilon = 1e-8  # Epsilon para evitar la división por cero en la actualización
        self.t = 0  # Contador de pasos

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        self.t += 1  # Incrementa el contador de pasos

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Actualización de los momentos m y v
        self.m = [(self.beta1 * m) + ((1 - self.beta1) * nb / len(mini_batch))
                  for m, nb in zip(self.m, nabla_b)]
        self.v = [(self.beta2 * v) + ((1 - self.beta2) * (nb / len(mini_batch))**2)
                  for v, nb in zip(self.v, nabla_b)]

        # Corrección de sesgo para los momentos m y v
        m_hat = [m / (1 - self.beta1**self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2**self.t) for v in self.v]

        # Actualización de pesos y sesgos con Adam
        self.weights = [w - (eta / (np.sqrt(vh) + self.epsilon)) * mh
                        for w, vh, mh in zip(self.weights, v_hat, m_hat)]
        self.biases = [b - (eta / (np.sqrt(vh) + self.epsilon)) * mh
                       for b, vh, mh in zip(self.biases, v_hat, m_hat)]

    # El resto del código permanece sin cambios

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete!!".format(j))

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
