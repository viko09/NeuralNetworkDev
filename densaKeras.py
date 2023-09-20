# Importamos librerias
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

# Cargamos los datos
dataset = mnist.load_data()

# Separamos datos en entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = dataset

# Mostramos las dimensiones de nuestros datos
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# Preprocesamos nuestros datos
x_trainv = x_train.reshape(60000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_test.reshape(10000, 784)
x_testv = x_testv.astype('float32')

# El numero de clases es para decidir el numero de entradas de nuestra
# matriz binaria
num_classes = 10
# Converts a class vector (integers) to binary class matrix.
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

# Paso 1: Definir el modelo de la red neuronal
model = Sequential()
# Capa de entrada
model.add(Dense(64, activation='relu', input_shape=(784,)))
# Capa oculta
model.add(Dense(32, activation='relu'))
# Capa de salida
model.add(Dense(num_classes, activation='sigmoid'))

