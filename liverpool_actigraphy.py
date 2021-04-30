"""
Alex Casson

Versions
07.04.20 - v1 - initial script

Aim
Load and process accelerometry data for sleep stages

Questions
fs?
units?
timestamp?
periodic interference?
"""

# %% Initalise Python
import math
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42 # for vecotr fonts in plots
matplotlib.rcParams['ps.fonttype'] = 42 # for vecotr fonts in plots
plt.close('all')
import seaborn as sns
sns.set()


# %% Define functions for filtering data

# Generate filter co-efficients
def filter_coefficients(cutoff, fs, order, ftype):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=ftype, analog=False)
    return b, a

# Filter data using filtfilt command
def data_filter(data, cutoff, fs, order, ftype):
    b, a = filter_coefficients(cutoff, fs, order, ftype)
    signal_filtered = filtfilt(b, a, data)
    return signal_filtered


# %% Load and format data

# Load data and extract overall acceleration magnitude
#acc_file = "./data_test.csv"
#acc_file = "./data_07.04.csv"
#acc_file = "./data_14.04.csv"
acc_file = "./data_18.04.csv"    
fs = 200
df = pd.read_csv(acc_file)
acc = df.to_numpy()
x = acc[:,0] / 9.806 # convert to g
y = acc[:,1] / 9.806 # convert to g
z = acc[:,2] / 9.806 # convert to g
ac = np.sqrt(x**2 + y**2 + z**2) - 1

# Filter data 
order = 2
#cutoffs = np.array([0.1, 20]) # high pass, low pass cut-offs
#a = data_filter(ac, cutoffs, fs, order, 'bandpass')
cutoffs = np.array([20]) # high pass, low pass cut-offs
a = data_filter(ac, cutoffs, fs, order, 'lowpass')
#plt.plot(time,a)

# Make time vector
time = np.arange(0,(np.size(a)/fs),1/fs)


# %% Calculate Cole-Kripke algorithm 

# Calculate the max activity present in each minute, divided as 10s windows with overlap 
window = 10
step = 2
duration_in_minutes = math.floor((len(a)/fs)/60)
A = np.full(duration_in_minutes, np.nan)
for i in range(duration_in_minutes):
    e_start = i * 60 * fs
    e_stop  = ((i+1) * 60 * fs) - 1
    #print(e_start, e_stop)
    a_minute = a[e_start:e_stop]
    
    # Extract 10s windows in each one minute block
    epochs_in_one_minute = int((60-window)/step) + 1
    activity = np.full(epochs_in_one_minute, np.nan)
    for j in range(epochs_in_one_minute):
        f_start = (j * step) * fs
        f_stop = (((j * step) + window) * fs) - 1
        #print(f_start, f_stop)
        epoch = a_minute[f_start:f_stop]
        # Count zero crossings in each 10s windowDrop out rate
        # 0.7, 0.5, 0.4, 0.3, and 0.2 drop rates were tested.
        #
        # Test accuracy was 59.79 % at 0.2 drop out and 58.76% at 0.4 drop out. To choose the best drop out among these two, their loss values were compared.
        #
        # Figure x: 0.2 drop out loss (left) and 0.4 drop out loss (right)
        # Although their validation loss were quite similar,  0.2 drop out was more stable and lower than 0.4 drop out. Therefore 0.2 drop out was chosen.
        activity[j] = sum(1 for i in range(1, len(epoch)) if epoch[i-1]*epoch[i]<0) 
    
    A[i] = np.max(activity)

# Apply Cole-Kripke algorithm 
# https://academic.oup.com/sleep/article/15/5/461/2749332
D = np.full(duration_in_minutes, np.nan)    
for k in range(4,duration_in_minutes-2):
    D[k] = 0.00001 * (404*A[k-4] + 598*A[k-3] + 326*A[k-2] + 441*A[k-1] + 1408*A[k] + 508*A[k+1] + 350*A[k+2])

    
# Threshold to determine sleep/wake
wake = np.full(duration_in_minutes, np.nan)
with np.errstate(invalid='ignore'): # suppress np warning due to nans
    wake[np.argwhere(D<1)] = False
    wake[np.argwhere(D>=1)] = True
