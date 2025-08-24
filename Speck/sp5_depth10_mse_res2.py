import Speck as sp
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
import matplotlib.pyplot as plt

def cyclic_lr(num_epochs, high_lr, low_lr):#алгоритм изменения скорости обучения
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_net(num_filters=32, num_outputs=1, d1=64, d2=64, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
    #построение сети
    #инпут - (размер батча, длина слова 32 бит, 2 канала - слово 1 и слово 2)
    inp = Input(shape=(64,))
    rs = Reshape((4, 16))(inp)
    perm = Permute((2, 1))(rs)
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    # add residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1=Add()([conv1,shortcut])
        tmp1=conv1
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2=Add()([conv2,tmp1])
        shortcut = Add()([shortcut, conv2])
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return (model)


key = 0x1918111009080100
speck = sp.Speck32_64(key,5)
X,Y=speck.make_data(10**7)
X_eval,Y_eval=speck.make_data(10**6)
net = make_net(depth=10, reg_param=10**-5)
#net.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
net.compile(optimizer='adam',loss='mse',metrics=['acc'])
net.summary()
lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001))
h = net.fit(X,Y,epochs=10,batch_size=10000,validation_data=(X_eval, Y_eval), callbacks=[lr])
# Обучение и проверка точности значений
plt.plot(h.history["acc"])
plt.plot(h.history["val_acc"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

# Обучение и проверка величины потерь
plt.plot(h.history["loss"])
plt.plot(h.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()





