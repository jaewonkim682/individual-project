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

fs = 350 # sample rate 
f = 1 # the frequency of the sine wave
x = np.arange(fs) 
# compute the value (amplitude) of the sin wave at the for each sample
y1 = np.sin(2*np.pi*(f*1)* (x/fs)) 
y2 = np.sin(2*np.pi*(f*1)* (x/fs))*0.3
y3 = np.sin(2*np.pi*(f*1)* (x/fs))*0.6
y4 = np.sin(2*np.pi*(f*2)* (x/fs))
#y5 = np.sin(2*np.pi*(f*5)* (x/fs))
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
#plt.plot(x,y5)
print(len(y1))
#print(y1)
