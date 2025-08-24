from Des import DES
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.backend import round

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
  des=DES(key,3)
  y=np.array([des.encrypt(i) for i in x])
  X=np.array([int.from_bytes(i,byteorder='big') for i in x])
  Y=np.array([int(i[0]) for i in y])
  return np.array(convert_to_binary(X)),np.array(Y)


def cyclic_lr(num_epochs, high_lr, low_lr):#алгоритм изменения скорости обучения
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_net(reg_param=0.0001, final_activation='sigmoid'):
    #построение сети
    inp = Input(shape=(64,))
    dense0 = Dense(64, kernel_regularizer=l2(reg_param))(inp)
    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('relu')(dense0)
    dense1 = Dense(64, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(64, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(1, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return (model)


inp=open('BelT.bin','rb')
X,Y=make_data(200000,inp)
X_eval,Y_eval=make_data(10000,inp)
X_test,Y_test=make_data(10000,inp)
X_pred,Y_pred=make_data(10,inp)
net = make_net(reg_param=10**-5)
net.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
net.summary()
lr = LearningRateScheduler(cyclic_lr(40,0.001, 0.0005))
h = net.fit(X,Y,epochs=40,batch_size=10000,validation_data=(X_eval, Y_eval), callbacks=lr)
# Обучение и проверка точности значений
plt.plot(h.history["accuracy"])
plt.plot(h.history["val_accuracy"])
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


results = net.evaluate(X_test, Y_test, batch_size=128)
print(results)
predictions=net.predict(X_pred)
print(round(predictions),Y_pred)