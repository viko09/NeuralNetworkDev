import netw
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = netw.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
