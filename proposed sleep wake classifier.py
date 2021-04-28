import math
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42 # for vecotr fonts in plots, imported code
matplotlib.rcParams['ps.fonttype'] = 42 # for vecotr fonts in plots, imported code
plt.close('all')
import seaborn as sns # making statistical graphics, imported code
sns.set()


# Generate filter co-efficients, imported code
def filter_coefficients(cutoff, fs, order, ftype):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=ftype, analog=False)
    return b, a

# Filter data using filtfilt command, imported code
def data_filter(data, cutoff, fs, order, ftype):
    b, a = filter_coefficients(cutoff, fs, order, ftype)
    signal_filtered = filtfilt(b, a, data)
    return signal_filtered
  
 
# Load acceleration and PSG data

#read acceleration data
fs = 50 #sampling frequency
df = pd.read_csv(acc_file)
acc = df.to_numpy()
x = acc[:,1] 
y = acc[:,2]
z = acc[:,3] 
ac = np.sqrt(x**2 + y**2 + z**2) - 1 

# Generate lowpass filter data, cutoff freqeucy is 20Hz, imported code
order = 2
cutoffs = np.array([20]) 
a = data_filter(ac, cutoffs, fs, order, 'lowpass')
time = np.arange(0,(np.size(a)/fs),1/fs)

# Calculate the max activity score in each 30-second  
window = 2
step = 1
duration_in_30sec = math.floor((len(a)/fs)/60)*2
A = np.full(duration_in_30sec, np.nan)
activity = np.full(duration_in_30sec, np.nan)
for i in range(duration_in_30sec):
    e_start = i * 30 * fs
    e_stop  = ((i+1) * 30 * fs) - 1
    epoch_30s = a[e_start:e_stop]
    epochs_in_one_30s = int((30-window)/step) + 1
    activity = np.full(epochs_in_one_30s, np.nan)
    for j in range(epochs_in_one_30s):
        f_start = (j * step) * fs
        f_stop = (((j * step) + window) * fs) - 1
        epoch = epoch_30s[f_start:f_stop]
       # Count zero crossings in each 2s window
        activity[j] = sum(1 for i in range(1, len(epoch)) if epoch[i-1]*epoch[i]<0)
    A[i] = np.max(activity)
    
# Apply Cole-Kripke algorithm 
D = np.full(duration_in_30sec, np.nan)    
for k in range(4,duration_in_30sec-2):
    D[k] = 0.0001 * (50*A[k-4] + 30*A[k-3] + 14*A[k-2] + 28*A[k-1] + 121*A[k] + 8*A[k+1] + 50*A[k+2])
    
# Threshold to determine sleep/wake of acceleration data, imported code
wake = np.full(duration_in_30sec, np.nan)
with np.errstate(invalid='ignore'): 
    wake[np.argwhere(D<1)] = False #sleep
    wake[np.argwhere(D>=1)] = True #wake

# read PSG data   
of = pd.read_csv(PSG_file) 
PSG = of.to_numpy()
PSG_data = PSG[:,1]

#Threshold to determine sleep/wake of PSG data
PSG1 = np.full(duration_in_30sec, np.nan)
with np.errstate(invalid='ignore'): 
    PSG1[np.argwhere(PSG_data<=0)] = True #wake
    PSG1[np.argwhere(PSG_data>0)] = False #sleep
    
#plot predictions and PSG    
plt.plot(PSG1)
plt.plot(wake)

#calculate specificity 
i = 0
Correct_wake = 0
Wrong_wake =0
k = 0
while i<len(PSG_data): 
    if PSG1[i] == 1:
        if wake[i] == PSG1[i]:
            Correct_wake = Correct_wake + 1 
        elif wake[i] != PSG1[i]:
            Wrong_wake = Wrong_wake +1
        k = k+1
    i = i+1
print(Correct_wake, Wrong_wake)
print((Correct_wake/(Correct_wake+Wrong_wake))*100)
print((Correct_wake/k)*100)

#Calculate sensitivity 
i = 0
Correct_sleep = 0
Wrong_sleep =0
while i<len(PSG_data): 
    if PSG1[i] == 0:
        if wake[i] == PSG1[i]:
            Correct_sleep = Correct_sleep+1 
        elif wake[i] != PSG1[i]:
            Wrong_sleep = Wrong_sleep +1 
    i = i+1
print(Correct_sleep, Wrong_sleep)
print((Correct_sleep/(Correct_sleep+Wrong_sleep))*100)


#Calculate overall accuracy
i = 0
Correct = 0
Wrong =0
while i<len(PSG_data):
    if wake[i] == PSG1[i]:
        Correct = Correct+1
    elif wake[i] != PSG1[i]:
        Wrong = Wrong +1
    i = i+1
print(Correct, Wrong)
print((Correct/(Correct+Wrong))*100)
