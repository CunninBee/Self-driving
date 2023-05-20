import matplotlib.pyplot as plt

print('SETTING UP')
import os
os.environ['TF_CP_MIN_LOG_LEVEL'] = '3'
from utlis import *
from sklearn.model_selection import train_test_split

#### STEP1
path = 'myData'
data = importDataInfo(path)

#### Step 2 (visualization and balancing of data)
data = balanceData(data,display=False)

#### Step 3 (formation of different arrays)
imagesPath, steerings = loadData(path,data)
#print(imagesPath[0],steering[0])

#### Step 4 (splitting of data into training and validation)
#training = data used while training of model
#validation data = data used to check performance of created model
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#### Step 5 (data augmentation)


#### STEP 6 (PREPROCESSING)


#### STEP 7 (BATCH GENERATOR)


#### STEP 8 (CREATING MODEL)
model = createModel()
model.summary()

#### STEP 9 (MODEL TRAINING)
history = model.fit(batchGen(xTrain,yTrain,100,1), steps_per_epoch=300, epochs=10,
                   validation_data=batchGen(xVal,yVal,100,0), validation_steps=200)

## Step 10 (Saving Model)
model.save("model.h5")
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()