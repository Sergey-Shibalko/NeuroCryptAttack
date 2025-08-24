from Des import DES
import numpy as np
from tensorflow import keras
from os import urandom
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Input, concatenate,  BatchNormalization, Activation
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.backend import abs,mean,round
import tensorflow as tf
#from keras import ops

def convert_to_binary(arr):
  res=[]
  for a in arr:
    tmp=[]
    for i in range(64):
      tmp.append(a % 2)
      a //= 2
    res.append(tmp[::-1])
  return res

def make_data(data_len,inp):
  key=inp.read(8)
  x=[inp.read(8) for _ in range(data_len)]
  des=DES(key,4)
  y=np.array([des.encrypt(i) for i in x])
  X=np.array([int.from_bytes(i,byteorder='big') for i in x])
  Y=np.array([int(i,2) for i in y])
  return np.array(convert_to_binary(X)),np.array(convert_to_binary(Y))

@keras.utils.register_keras_serializable(name="hamming_metric")
def hamming_metric(y_true, y_pred):
    hamming = abs(y_true - round(y_pred))
    return mean(hamming)

def cyclic_lr(num_epochs, high_lr, low_lr):#алгоритм изменения скорости обучения
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_net(reg_param=0.0001, final_activation='sigmoid'):
    #построение сети
    #инпут - (размер батча, длина слова 32 бит, 2 канала - слово 1 и слово 2)
    inp = Input(shape=(64,))
    #rs = Reshape((4, 16))(inp)
    #perm = Permute((2, 1))(rs)
    #flat1 = Flatten()(shortcut)
    dense0 = Dense(64, kernel_regularizer=l2(reg_param))(inp)
    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('sigmoid')(dense0)
    inp1=concatenate([inp,dense0],axis=1)
    dense1 = Dense(128, kernel_regularizer=l2(reg_param))(inp1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('sigmoid')(dense1)
    inp2=concatenate([inp1,dense1],axis=1)
    dense2 = Dense(128, kernel_regularizer=l2(reg_param))(inp2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('sigmoid')(dense2)
    inp3 = concatenate([inp2, dense2], axis=1)
    out = Dense(64, activation=final_activation, kernel_regularizer=l2(reg_param))(inp3)
    model = Model(inputs=inp, outputs=out)
    return (model)


inp=open('BelT.bin','rb')
X,Y=make_data(1000000,inp)
X_eval,Y_eval=make_data(1000,inp)
X_test,Y_test=make_data(1000,inp)
net = make_net(reg_param=10**-5)
net.compile(optimizer='adam',loss='binary_crossentropy',metrics=[hamming_metric])
#net.compile(optimizer='adam',loss='mse',metrics=[hamming_metric])
net.summary()
lr = LearningRateScheduler(cyclic_lr(40,0.0005, 0.0001))
#h = net.fit(X,Y,epochs=20,batch_size=10000,validation_data=(X_eval, Y_eval), callbacks=lr)
h = net.fit(Y,X,epochs=40,batch_size=10000,validation_data=(Y_eval, X_eval), callbacks=lr)
net.save('DES_cascade_64_128_128_64_BinCros.keras')
# Обучение и проверка точности значений
#plt.plot(h.history["acc"])
#plt.plot(h.history["val_acc"])
#plt.title("Model accuracy")
#plt.ylabel("Accuracy")
#plt.xlabel("Epoch")
#plt.legend(["Train", "Test"], loc="upper left")
#plt.show()

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

results = net.evaluate(Y_test, X_test, batch_size=128)
print(results)
X_pred,Y_pred=make_data(10,inp)
predictions=net.predict(X_pred)
print(predictions,Y_pred)
