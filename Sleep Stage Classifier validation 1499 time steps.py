import math
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42 # For vecotr fonts in plots, imported code
matplotlib.rcParams['ps.fonttype'] = 42 # For vecotr fonts in plots, imported code
plt.close('all')
import seaborn as sns # Making statistical graphics, imported code
sns.set()

# Function for plotting accuracy and loss graphs, imported code
def plot_graphs(history,metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.title('Training and Validation '+metric.capitalize()) #uppercase metric?
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric,'val_'+metric])
    plt.show()
    
  
# Function for plotting the confusion matrix, imported code    
def plot_CM(true_labels, predictions, activities): 
    max_true = np.argmax(true_labels, axis = 1)
    max_prediction = np.argmax(predictions, axis = 1)
    CM = confusion_matrix(max_true, max_prediction)
    plt.figure(figsize=(16,14))
    sns.heatmap(CM, xticklabels = activities, yticklabels = activities,
                annot = True, fmt = 'd',cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

fs = 350 # Sample rate 
f = 1 # Frequency of the sine wave
x = np.arange(fs) 

# Assign the amplitude of each sine wave 
y1 = np.sin(2*np.pi*(f*1)* (x/fs)) 
y2 = np.sin(2*np.pi*(f*1)* (x/fs))*0.3
y3 = np.sin(2*np.pi*(f*1)* (x/fs))*0.6
y4 = np.sin(2*np.pi*(f*2)* (x/fs))

# Generate the fundamental wave of the simple input data
y = np.empty((50,350))
y[:]=np.nan 
for i in range (50):
    if 0<=i <7:
        y[i] = y1
    elif 7<=i<9:
        y[i] = y2
    elif 9<=i <34:
        y[i] = 0
    elif 34<=i<41:
        y[i] = y3
    else:
        y[i]= 0.25
plt.plot(y)
print(y.shape)

# Concatenate the fundamental wave
Y = np.empty((970,350))
Y[:]=np.nan 
Y = np.concatenate((y, y))
for i in range(17):
    Y = np.concatenate((Y, y))
Y = np.concatenate((Y, y[0:18]))
print(Y.shape)
plt.plot(Y)

# Copy the y data and paste it to x and z        
X = Y
Z = Y

# Assign class according to x, y, and z data
sub_PSG_data = np.array([0,0,0,0,0,0,0,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4])

#  Concatenate the PSG of the simple input data
PSG_data = np.empty((968))
PSG_data[:]=np.nan 
PSG_data = np.concatenate((sub_PSG_data, sub_PSG_data))
for i in range(17):
    PSG_data = np.concatenate((PSG_data, sub_PSG_data))
PSG_data = np.concatenate((PSG_data, sub_PSG_data[0:18]))

# Count each class in the PSG data
classes, counts = np.unique(PSG_data, return_counts=True)
dict(zip(classes, counts))

# Divide x, y, z, and PSG into training and test dataset
xtrain_acc, xtest_acc, ytrain_acc, ytest_acc, ztrain_acc, ztest_acc, psg_train, psg_test = train_test_split(
 X, Y, Z, PSG_data, test_size=0.3,random_state=20, shuffle=True)

# Rearrange training and test dataset for LSTM, imported code
train_set = [xtrain_acc, ytrain_acc, ztrain_acc]
train_set = np.array(np.dstack(train_set),dtype=np.float32)
test_set = [xtest_acc, ytest_acc, ztest_acc]
test_set = np.array(np.dstack(test_set),dtype = np.float32)

# Apply one-hot label encoding to the PSG data, imported code
psg_train = tf.keras.utils.to_categorical(psg_train)
psg_test = tf.keras.utils.to_categorical(psg_test)

# Build LSTM Model
time_steps = train_set.shape[1] #imported code
features = train_set.shape[2] #imported code
model = tf.keras.Sequential() #imported code
model.add(tf.keras.layers.LSTM(32, input_shape = (time_steps,features),return_sequences=True))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(5, activation ='softmax'))

# Optimizer and loss function for the LSTM model, imported code except the learning rate
model.compile(optimizer = tf.keras.optimizers.Adam(0.002), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the LSTM Model
history = model.fit(train_set, psg_train, epochs = 38, batch_size = 8,
          validation_split = 0.3, shuffle = True)

# Testing the LSTM Model
test_loss, test_acc = model.evaluate(test_set, psg_test, batch_size = 8)
model.summary()

# Plot accuracy and loss graphs, imported code
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

# Plot the confusion matrix of the test predictions, imported code
predictions = model.predict(test_set)
Activities = ['0', '1', '2', '3', '4']
plot_CM(psg_test,predictions, Activities)

# Display the overall accuracy, imported code
print('\nTest accuracy: ', test_acc)
