import math
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42 # For vecotr fonts in plots, imported code1
matplotlib.rcParams['ps.fonttype'] = 42 # For vecotr fonts in plots, imported code1
plt.close('all')
import seaborn as sns # Make statistical graphics, imported code1
sns.set()


# Generate filter co-efficients, imported code1
def filter_coefficients(cutoff, fs, order, ftype):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=ftype, analog=False)
    return b, a

# Filter data using filtfilt command, imported code1
def data_filter(data, cutoff, fs, order, ftype):
    b, a = filter_coefficients(cutoff, fs, order, ftype)
    signal_filtered = filtfilt(b, a, data)
    return signal_filtered
  
PSG_file = # Load PSG data
acc_file = # Load acceleration data

# Read the acceleration data
fs = 50*4 #Sampling frequency
df = pd.read_csv(acc_file)
acc = df.to_numpy()
x = acc[:,1]
y = acc[:,2]
z = acc[:,3]

# Upsample the acceleration data
time = np.linspace(0, len(x)-1, len(x))
xvals = np.linspace(0, len(x)-1, len(x)*4)
yvals = np.linspace(0, len(y)-1, len(y)*4)
zvals = np.linspace(0, len(z)-1, len(z)*4)
x = np.interp(xvals, time, x)
y = np.interp(yvals, time, y)
z = np.interp(zvals, time, z)
ac = np.sqrt(x**2 + y**2 + z**2) - 1

# Generate lowpass filter, cutoff freqeucy is 20Hz, imported code1
order = 2
cutoffs = np.array([20]) 
a = data_filter(ac, cutoffs, fs, order, 'lowpass')


# The maximum 10-second nonoverlapping epoch of each minute
#window = 10
#step = 10
#duration_in_minutes = math.floor((len(a)/fs)/60)
#A = np.full(duration_in_minutes, np.nan)
#for i in range(duration_in_minutes):
#    e_start = i * 60 * fs
#    e_stop  = ((i+1) * 60 * fs) - 1
#    a_minute = a[e_start:e_stop]
    
    
#    epochs_in_one_minute = int((60-window)/step) + 1
#    activity = np.full(epochs_in_one_minute, np.nan)
#    for j in range(epochs_in_one_minute):
#        f_start = (j * 10) * fs
#        f_stop = (((j +1) + 10) * fs) - 1
#        epoch = a_minute[f_start:f_stop]
        # Count zero crossings in each 10s window
 #       activity[j] = sum(1 for i in range(1, len(epoch)) if epoch[i-1]*epoch[i]<0) 
    
 #   A[i] = np.max(activity) 
    #print(activity)
#print(epochs_in_one_minute)
#plt.plot(A)

# The maximum 30-second nonoverlapping epoch of each minute
#duration_in_minutes = math.floor((len(a)/fs)/60)
#A = np.full(duration_in_minutes, np.nan)
#for i in range(duration_in_minutes):
#    e_start = i * 60 * fs
#    e_stop  = ((i+1) * 60 * fs) - 1
#    a_minute = a[e_start:e_stop]
    
    
#    epochs_in_one_minute = int((60-window)/step) + 1
#    activity = np.full(epochs_in_one_minute, np.nan)
#    for j in range(epochs_in_one_minute):
#        f_start = (j * 30) * fs
#        f_stop = (((j + 1) * 30) * fs) - 1
#        epoch = a_minute[f_start:f_stop]
        # Count zero crossings in each 10s window
#        activity[j] = sum(1 for i in range(1, len(epoch)) if epoch[i-1]*epoch[i]<0) 
    
 #   A[i] = np.max(activity) 
#print(epochs_in_one_minute)
#plt.plot(A)

# The mean activity of each minute
duration_in_minutes = math.floor((len(a)/fs)/60)#fs= no of samples/ sampling time,
A = np.full(duration_in_minutes, np.nan)
for i in range(duration_in_minutes):
    e_start = i * 60 * fs
    e_stop  = ((i+1) * 60 * fs) - 1
    epoch = a[e_start:e_stop]
    A[i] = (sum(1 for i in range(1, len(epoch)) if epoch[i-1]*epoch[i]<0))/(len(epoch)-1)
plt.plot(A)
print(duration_in_minutes)

# The maximum 10-second overlapping epoch of each minute
#window = 10
#step = 2
#duration_in_minutes = math.floor((len(a)/fs)/60)
#A = np.full(duration_in_minutes, np.nan)
#for i in range(duration_in_minutes):
#    e_start = i * 60 * fs
#    e_stop  = ((i+1) * 60 * fs) - 1
#    a_minute = a[e_start:e_stop]
   
    
#    epochs_in_one_minute = int((60-window)/step) + 1
#    activity = np.full(epochs_in_one_minute, np.nan)
#    for j in range(epochs_in_one_minute):
#        f_start = (j * step) * fs
#        f_stop = (((j * step) + window) * fs) - 1
#        epoch = a_minute[f_start:f_stop]
        # Count zero crossings in each 10s window
#        activity[j] = sum(1 for i in range(1, len(epoch)) if epoch[i-1]*epoch[i]<0) 
    
#    A[i] = np.max(activity) 
    #print(activity)
#print(epochs_in_one_minute)
#plt.plot(A)

# Apply Cole-Kripke algorithm 
D = np.full(duration_in_minutes, np.nan)    
for k in range(4,duration_in_minutes-2):
    #D[k] = 0.00001 * (404*A[k-4] + 598*A[k-3] + 326*A[k-2] + 441*A[k-1] + 1408*A[k] + 508*A[k+1] + 350*A[k+2]) # For the maximum 10-second overlapping epoch of each minute
    D[k] = 0.001 * (106*A[k-4] + 54*A[k-3] + 58*A[k-2] + 76*A[k-1] + 230*A[k] + 74*A[k+1] + 67*A[k+2]) # For the mean activity of each minute
    #D[k] = 0.0001 * (50*A[k-4] + 30*A[k-3] + 14*A[k-2] + 28*A[k-1] + 121*A[k] + 8*A[k+1] + 50*A[k+2]) # For the maximum 30-second nonoverlapping epoch of each minute
    #D[k] = 0.00001 * (550*A[k-4] + 378*A[k-3] + 413*A[k-2] + 699*A[k-1] + 1736*A[k] + 287*A[k+1] + 309*A[k+2]) # For the maximum 10-second nonoverlapping epoch of each minute

    
# Threshold to determine sleep/wake of the acceleration data, imported code1
wake = np.full(duration_in_minutes, np.nan)
with np.errstate(invalid='ignore'): # suppress np warning due to nans
    wake[np.argwhere(D<1)] = False #sleep
    wake[np.argwhere(D>=1)] = True #wake

# Read the PSG data  
of = pd.read_csv(PSG_file) 
PSG = of.to_numpy()
PSG_data = PSG[:,1]
plt.plot(PSG_data)

# Resample the PSG data
half = int(len(PSG_data)/2)
E = np.full(len(PSG_data), np.nan)    
for k in range(0,half):
       E[k] = PSG_data[k*2]
plt.plot(E)

# Threshold to determine sleep/wake of the PSG data, imported code1
PSG1 = np.full(len(PSG_data), np.nan)
with np.errstate(invalid='ignore'): # suppress np warning due to nans
    PSG1[np.argwhere(E<=0)] = True #wake
    PSG1[np.argwhere(E>0)] = False #sleep
    
# Plot predictions and PSG
fig, ax = plt.subplots()
ax.plot(PSG1, label='PSG')
ax.plot(wake,label='Prediction')
leg = ax.legend(bbox_to_anchor = [0.5, 0.2])

# Calculate the overall accuracy
i = 0
Correct = 0
Wrong =0
while i<half:
    if wake[i] == PSG1[i]:
        Correct = Correct+1
    elif wake[i] != PSG1[i]:
        Wrong = Wrong +1
    i = i+1
print(Correct, Wrong)
print((Correct/(Correct+Wrong))*100)
