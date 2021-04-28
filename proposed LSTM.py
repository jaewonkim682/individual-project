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
import seaborn as sns # Make statistical graphics, imported code
sns.set()

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
    
# Function for plotting accuracy and loss graphs, imported code
def plot_graphs(history,metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric],'')
    plt.title('Training and Validation '+metric.capitalize()) #uppercase metric?
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric,'val_'+metric])
    plt.show()
    
PSG_file = # Load PSG data
acc_file = # Load acceleration data

# Read acceleration and PSG data    
fs = 50
df = pd.read_csv(acc_file)
acc = df.to_numpy()
x = acc[:,1] 
y = acc[:,2]
z = acc[:,3]
of = pd.read_csv(PSG_file) 
PSG = of.to_numpy()
PSG_data = PSG[:,1]
time = len(PSG_data)

# Apply min-max normalization
scaler = MinMaxScaler(feature_range=(-1,1))
x = scaler.fit_transform(x.reshape(-1,1)).reshape(x.shape)
y = scaler.fit_transform(y.reshape(-1,1)).reshape(y.shape)
z = scaler.fit_transform(z.reshape(-1,1)).reshape(z.shape)

# Separate x data into 30-second intervals
duration_in_minutes = math.floor((len(x)/fs)/30)
epoch_length = 1499 #time steps
X = np.empty((time,epoch_length))
X[:]=np.nan
for i in range(time): 
    e_start = i * 30 * fs
    e_stop  = ((i+1) * 30 * fs) - 1
    epoch = x[e_start:e_stop]
    X[i] = epoch


# Separate y data into 30-second intervals
duration_in_minutes = math.floor((len(y)/fs)/30)
Y = np.empty((time,epoch_length))
Y[:]=np.nan
for i in range(time):
    e_start = i * 30 * fs
    e_stop  = ((i+1) * 30 * fs) - 1
    epoch = y[e_start:e_stop]
    Y[i] = epoch


# Separate z data into 30-second intervals
duration_in_minutes = math.floor((len(z)/fs)/30)
Z = np.empty((time,epoch_length))
Z[:]=np.nan
for i in range(time): 
    e_start = i * 30 * fs
    e_stop  = ((i+1) * 30 * fs) - 1
    epoch = z[e_start:e_stop]
    Z[i] = epoch


# Edit PSG data
for x in range(len(PSG_data)): 
    if PSG_data[x] == -1:
        PSG_data[x] = 0
    elif PSG_data[x] == 5:
        PSG_data[x] = 4


# Divide acceleration and PSG data into training and test data
xtrain_acc, xtest_acc, ytrain_acc, ytest_acc, ztrain_acc, ztest_acc, psg_train, psg_test = train_test_split(
 X, Y, Z, PSG_data, test_size=0.3,random_state=20, shuffle=True)
 
# Generate train and test sets for LSTM, imported code
train_set = [xtrain_acc, ytrain_acc, ztrain_acc]
train_set = np.array(np.dstack(train_set),dtype=np.float32)
test_set = [xtest_acc, ytest_acc, ztest_acc]
test_set = np.array(np.dstack(test_set),dtype = np.float32)


# One-hot label encoding to the PSG data, imported code
psg_train = tf.keras.utils.to_categorical(psg_train)
psg_test = tf.keras.utils.to_categorical(psg_test)


# Build LSTM Model
time_steps = train_set.shape[1] # imported code
features = train_set.shape[2] # imported code
model = tf.keras.Sequential() # imported code
model.add(tf.keras.layers.LSTM(32, input_shape = (time_steps,features),return_sequences=True))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(5, activation ='softmax'))

# Optimizer and loss function, imported code except the learning rate
model.compile(optimizer = tf.keras.optimizers.Adam(0.002), loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(train_set, psg_train, epochs = 38, batch_size = 8,
          validation_split = 0.3, shuffle = True)

# Model testing
test_loss, test_acc = model.evaluate(test_set, psg_test, batch_size = 8)
model.summary()

# Plot accuracy and loss graphs, imported code
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

# Plot the confusion matrix of the test predictions, imported code
predictions = model.predict(test_set)
print(predictions.shape)
print(psg_test.shape)
Activities = ['Wake', '1', '2','3', '5'] # 1 is N1, 2 is N2, 3 is N3, and 5 is REM 
plot_CM(psg_test,predictions, Activities)

# Display the overall accuracy, imported code
print('\nTest accuracy: ', test_acc)
