import pandas as pd
import numpy as np
import sklearn.model_selection
from tensorflow import keras
import matplotlib.pyplot as plt

df = pd.read_csv(r"data.csv")
df.replace("?",-99999,inplace=True)
df = df.drop(['index','token_id','index','token_id','Background','Clothes','Earring','Eyes','Fur','Hat','Mouth','num_sales','image_url'],1)


#removing outliers (values greater than 700)
df = df[df['price']<700]
data = np.array(df)

"""Structure of rows in array
0 - price (output)
1 - rarity score
2 - trait_count
3 - trait_count_rarity
4 - background Rarity
5 - Clothes rarity
6 - Earring  Rarity
7 - Eyes rarity
8 - Fur Rarity
9 - Hat Rarity
10 - Mouth rarity


NOTE: All the "missing rarities" should be the last new columns of the list 


11 - Earring Missing Rarity
12 - Clothes missing Rarity
13 - Hat missing Rarity
"""


index = 0
count = -1


priceindex = 0
rarityscoreindex = 1
traitcountindex = 2
traitcountrarityindex = 3
backgroundrarityindex = 4
clothesrarityindex = 5
earringrarityindex = 6
eyesrarityindex = 7
furrarityindex = 8
hatrarityindex = 9
mouthrarityindex = 10
earringmissingrarityindex = 11
clothesmissingrarityindex = 12
hatmissingrarityindex = 13

marked_for_removal_val = 100
for row in data:
    count += 1
    #print(row)
    index = 0
    for num in row:

        #if clothes are missing
        if np.isnan(num) and index == clothesrarityindex:

            clothes_missing_rarity = row[clothesmissingrarityindex]
            row[index] = clothes_missing_rarity
            row[clothesmissingrarityindex] = marked_for_removal_val
            data[count] = row


        #if hat is missing
        if np.isnan(num) and index == hatrarityindex:
            hat_missing_rarity = row[hatmissingrarityindex]
            row[index] = hat_missing_rarity
            row[hatmissingrarityindex] = marked_for_removal_val
            data[count] = row

        #if earrings are missing
        if np.isnan(num) and index == earringrarityindex:
            earring_missing_rarity = row[earringmissingrarityindex]
            row[index] = earring_missing_rarity
            row[earringmissingrarityindex] = marked_for_removal_val
            data[count] = row

        index += 1




#remove the "missing items" rows
data2 = data[:,0:11]

input_data = data2[:,1:]
output_data = data2[:,0]


x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(input_data,output_data,test_size=0.01)


print(len(x_train[0]))
print("max",max(y_train))
file = open("kfactorvalue.txt",'w')
file.write(str(max(y_train)))
file.close()

#normalizing data
y_train = y_train/(max(y_train))

#dnn training
print(max(y_train),min(y_train))
model = keras.Sequential([
keras.layers.Dense(32,input_shape = (10,),activation='relu'),
keras.layers.Dense(32,activation='relu'),
keras.layers.Dense(32,activation='relu'),
keras.layers.Dense(32,activation='relu'),
keras.layers.Dense(1,activation="sigmoid")])



model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=["mae"])

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,batch_size=1)
#test_acc = model.evaluate(x_test,y_test)
model.save("dnnmodel.h5")
