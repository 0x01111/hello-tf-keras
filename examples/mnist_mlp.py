'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.

Test loss: 0.11746364831924438
Test accuracy: 0.9837999939918518
82s每论
'''

from __future__ import print_function

import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

print(y_test[1:4])
# 激活函数
# 1. https://keras.io/zh/activations/
# 2.
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2)) # https://www.zhihu.com/question/61751133
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2)) # 0.9999 时候效果最差，相当于随机猜了
model.add(Dense(num_classes, activation='softmax'))

# tensorflow.keras.utils.plot_model(model, "./mnist_mlp_model.png", show_shapes=True)
# 优化器说明
# 1. https://zhuanlan.zhihu.com/p/34169434
# 2. https://keras.io/zh/optimizers/#adagrad
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.summary()

print(history.history)
from myplot import *
plot_curve(history)
