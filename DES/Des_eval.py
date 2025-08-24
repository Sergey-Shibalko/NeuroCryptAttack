from keras.models import load_model
from keras.backend import round,mean
import numpy as np
from Des import DES
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


def hamming_metric(y_true, y_pred):
  hamming = abs(y_true - round(y_pred))
  return mean(hamming)

file=input('Введите имя файла')
number=int(input("Введите количество слов"))
inp=open(file,'rb')
X_test,Y_test=make_data(number,inp)
net=load_model('DES_cascade_64_128_128_64_BinCros.keras',custom_objects={'Custom>hamming_metric': hamming_metric})
results = net.evaluate(Y_test, X_test, batch_size=512)
print(results)
