from keras.models import load_model
import numpy as np


file = open("kfactorvalue.txt",'r')
k = float(file.readlines()[0].strip())
file.close()
print(k)
model = load_model("dnnmodel.h5")
d = np.array([87.18096971,6,0.532353235,0.129112911,0.188519,0.088208821,0.171417142,0.135213521,0.02310231,0.227222722])

d = np.reshape(d,(1,10))
#print(d)
predictions = model.predict(d,batch_size=1)
print(predictions[0]*k)