from des import DesKey
import numpy as np
from os import urandom
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Conv1D, Input, Reshape,concatenate, Permute, Add, Flatten, BatchNormalization, Activation,AveragePooling1D
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras import ops

def convert_to_binary(arr):
  res=[]
  for a in arr:
    tmp=[]
    for i in range(64):
      tmp.append(a % 2)
      a //= 2
    res.append(tmp[::-1])
  return res

def make_data(data_len,file=''):
  if len(file)==0:
    x=np.array([urandom(8) for _ in range(data_len)])
    key=urandom(8)
  else:
    inp=open(file,'rb')
    key=inp.read(8)
    x=np.array([inp.read(8) for _ in range(data_len)])
  key0 = DesKey(key)
  y=np.array([key0.encrypt(i,padding=True) for i in x])
  X=np.array([int.from_bytes(i,byteorder='big') for i in x])
  Y=np.array([int.from_bytes(i,byteorder='big') for i in y])
  return np.array(convert_to_binary(X)),np.array(convert_to_binary(X))

def hamming_metric(y_true, y_pred):
  hamming=ops.absolute(y_true-y_pred)
  return ops.mean(hamming, axis=-1)

def cyclic_lr(num_epochs, high_lr, low_lr):#алгоритм изменения скорости обучения
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_net(num_filters=32, num_outputs=1, d1=64, d2=64, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
    #построение сети
    #инпут - (размер батча, длина слова 32 бит, 2 канала - слово 1 и слово 2)
    inp = Input(shape=(64,))
    #rs = Reshape((4, 16))(inp)
    #perm = Permute((2, 1))(rs)
    #flat1 = Flatten()(shortcut)
    dense0 = Dense(128, kernel_regularizer=l2(reg_param))(inp)
    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('relu')(dense0)
    dense1 = Dense(512, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(256, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    dense3 = Dense(128, kernel_regularizer=l2(reg_param))(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)
    out = Dense(64, activation=final_activation, kernel_regularizer=l2(reg_param))(dense3)
    model = Model(inputs=inp, outputs=out)
    return (model)



X,Y=make_data(10000,'/content/drive/MyDrive/BelT.bin')
X_eval,Y_eval=make_data(1000,'/content/drive/MyDrive/BelT.bin')
net = make_net(depth=10, reg_param=10**-5)
net.compile(optimizer='adam',loss='binary_crossentropy',metrics=[hamming_metric])
#net.compile(optimizer='adam',loss='mse',metrics=[hamming_metric])
net.summary()
lr = LearningRateScheduler(cyclic_lr(20,0.005, 0.0005))
#h = net.fit(X,Y,epochs=20,batch_size=10000,validation_data=(X_eval, Y_eval), callbacks=lr)
h = net.fit(Y,X,epochs=20,batch_size=100,validation_data=(Y_eval, X_eval), callbacks=lr)

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