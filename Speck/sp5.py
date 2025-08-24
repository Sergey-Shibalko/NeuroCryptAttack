import numpy as np
from os import urandom
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
import matplotlib.pyplot as plt

class Speck32_64(object):
    def __init__(self, key, num_rounds):  # инициализация класса, вычисляем массив ключей,
        self.key = key  # используемых для шифрования и дешифрования
        self.num_rounds=num_rounds
        l_schedule = [(key >> (x * 16)) % (1 << 16) for x in
                      range(1, 4)]
        self.key_schedule = []
        k2 = self.key % (1 << 16)
        self.key_schedule.append(k2)
        for i in range(self.num_rounds - 1):
            k1, k2 = self.encrypt_round(l_schedule[i], k2, i)
            l_schedule.append(k1)
            self.key_schedule.append(k2)

    def make_key_schedule(self, key):
        l_schedule = [(key >> (x * 16)) % (1 << 16) for x in
                      range(1, 4)]
        key_schedule = []
        k2 = key % (1 << 16)
        key_schedule.append(k2)
        for i in range(self.num_rounds - 1):
            k1, k2 = self.encrypt_round(l_schedule[i], k2, i)
            l_schedule.append(k1)
            key_schedule.append(k2)
        return key_schedule


    def encrypt_round(self, x, y, k):  # один раунд шифрования
        # Циклический сдвиг первого слова вправо на 7 бит;
        # Сложение второго слова с первым по модулю 2 в степени длины слова;
        # Операция XOR ключа и результата сложения;
        # Циклический сдвиг второго слова влево на 2 бита;
        # Операция XOR второго слова и результата предыдущего XOR.
        x = self.ROR(x, 7)
        x = (x + y) % (1 << 16)
        x = x ^ k
        y = self.ROL(y, 2)
        y = y ^ x
        return [x, y]

    def ROL(self, x, a):  # Циклический сдвиг вправо на 7 бит
        return ((x << a) + (x >> 16 - a)) % (1 << 16)

    def ROR(self, x, a):  # Циклический сдвиг влево на 2 бита;
        return ((x >> a) + (x << 16 - a)) % (1 << 16)

    def encrypt(self, plaintext,key_sc=[]):  # алгоритм шифрования
        # разбиваем слово на 2
        # повторяем раунд шифрования 22 раза
        if len(key_sc)==0:
            key_sc=self.key_schedule
        l = plaintext >> 16
        r = plaintext % (1 << 16)
        for i in key_sc:
            l, r = self.encrypt_round(l, r, i)
        return (l << 16) + r

    def decrypt(self, ciphertxt):  # алгоритм дешифрования
        # разбиваем слово на 2
        # повторяем раунд дешифрования 22 раза
        # используем ключи с конца
        l = ciphertxt >> 16
        r = ciphertxt % (1 << 16)
        for i in reversed(self.key_schedule):
            l, r = self.decrypt_round(l, r, i)
        return (l << 16) + r

    def decrypt_round(self, x, y, k):  # один раунд шифрования
        # Операция XOR второго и первого слова.
        # Циклический сдвиг второго слова вправо на 2 бита;
        # Операция XOR ключа и первого слова;
        # Разность первого слова со вторым по модулю 2 в степени длины слова;
        # Циклический сдвиг первого слова влево на 7 бит;
        y = y ^ x
        y = self.ROR(y, 2)
        x = x ^ k
        x = (x - y) % (1 << 16)
        x = self.ROL(x, 7)
        return [x, y]

    def convert_to_binary(self,arr):
        X = np.zeros((2 * 32, len(arr[0])), dtype=np.uint8)
        for i in range(2 * 32):
            index = i // 32
            offset = 32 - (i % 32) - 1
            X[i] = (arr[index] >> offset) & 1
        X = X.transpose()
        return (X)


def cyclic_lr(num_epochs, high_lr, low_lr):#алгоритм изменения скорости обучения
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_net(num_filters=32, num_outputs=1, d1=64, d2=64, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
    #построение сети
    #инпут - (размер батча, длина слова 32 бит, 2 канала - слово 1 и слово 2)
    inp = Input(shape=(1,))
    #rs = Reshape((4, 16))(inp)
    #perm = Permute((2, 1))(rs)
    flat1 = Flatten()(inp)
    dense1 = Dense(256, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(256, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    dense3 = Dense(64, kernel_regularizer=l2(reg_param))(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)
    out = Dense(num_outputs, activation='relu', kernel_regularizer=l2(reg_param))(dense3)
    model = Model(inputs=inp, outputs=out)
    return (model)


X=np.frombuffer(urandom(4*10000000), dtype=np.uint32)
key = 0x1918111009080100
speck = Speck32_64(key,2)
Y=speck.encrypt(X).tolist()
X_eval=np.frombuffer(urandom(4*100000), dtype=np.uint32)
Y_eval=speck.encrypt(X_eval).tolist()
net = make_net(depth=10, reg_param=10**-5)
#net.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
net.compile(optimizer='adam',loss='mse',metrics=['acc'])
net.summary()
lr = LearningRateScheduler(cyclic_lr(30,0.002, 0.0001))
h = net.fit(Y,X.tolist(),epochs=10,batch_size=10000,validation_data=( Y_eval,X_eval.tolist()), callbacks=lr)
# Обучение и проверка величины потерь
plt.plot(h.history["loss"])
plt.plot(h.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()
#print(speck.convert_to_binary(Y.tolist()))