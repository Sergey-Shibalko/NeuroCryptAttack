import numpy as np
from os import urandom
import sys
import numpy.random
import itertools
import Gost
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Dense,  Input, BatchNormalization, Activation
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.backend import abs,mean,round

def convert_to_binary(arr):
  res=[]
  for a in arr:
    tmp=[]
    for i in range(64):
      tmp.append(a % 2)
      a //= 2
    res.append(tmp[::-1])
  return res

def make_data(data_len,gost):

  X=np.frombuffer(urandom(8*data_len), dtype=np.int64)
  Y =np.array([gost.crypt(i) for i in X],dtype=np.int64)
  return np.array(convert_to_binary(X),dtype=np.float32),np.array(convert_to_binary(Y),dtype=np.float32)

def cyclic_lr(num_epochs, high_lr, low_lr):#алгоритм изменения скорости обучения
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def hamming_metric(y_true, y_pred):
  hamming=abs(y_true-round(y_pred))
  return mean(hamming, axis=-1)

def make_net(reg_param=0.0001, final_activation='sigmoid'):
    #построение сети
    #инпут - (размер батча, длина слова 32 бит, 2 канала - слово 1 и слово 2)
  inp = Input(shape=(64,))
  dense0 = Dense(64, kernel_regularizer=l2(reg_param))(inp)
  dense0 = BatchNormalization()(dense0)
  dense0 = Activation('relu')(dense0)
  dense3 = Dense(128, kernel_regularizer=l2(reg_param))(dense0)
  dense3 = BatchNormalization()(dense3)
  dense3 = Activation('relu')(dense3)
  #dense4 = Dense(256, kernel_regularizer=l2(reg_param))(dense3)
  #dense4 = BatchNormalization()(dense4)
  #dense4 = Activation('relu')(dense4)
  dense5 = Dense(128, kernel_regularizer=l2(reg_param))(dense3)
  dense5 = BatchNormalization()(dense5)
  dense5 = Activation('relu')(dense5)
  out = Dense(64, activation=final_activation, kernel_regularizer=l2(reg_param))(dense5)
  model = Model(inputs=inp, outputs=out)
  return (model)



a = Gost.Gost(3)
a.set_key(int.from_bytes(urandom(32),byteorder='big'))
X,Y=make_data(10000,a)
X_eval,Y_eval=make_data(1000,a)
X_test,Y_test=make_data(1000,a)
net = make_net(reg_param=10**-5)
net.compile(optimizer='adam',loss='binary_crossentropy',metrics=[hamming_metric])
#net.compile(optimizer='adam',loss='mse',metrics=[hamming_metric])
net.summary()
lr = LearningRateScheduler(cyclic_lr(20,0.0005, 0.0001))
#h = net.fit(X,Y,epochs=20,batch_size=10000,validation_data=(X_eval, Y_eval), callbacks=lr)
h = net.fit(Y,X,epochs=20,batch_size=1000,validation_data=(Y_eval, X_eval), callbacks=lr)

# Обучение и проверка величины потерь
plt.plot(h.history["loss"])
plt.plot(h.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

plt.plot(h.history["hamming_metric"])
plt.plot(h.history["val_hamming_metric"])
plt.title("Model hamming_metric")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

results = net.evaluate(Y_test, X_test, batch_size=512)
print(results)



