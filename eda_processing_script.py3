#!/usr/bin/env python3

import joblib
import pickle
import sys
from biosppy import signals
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# if len(sys.argv) != 2:
# 	print("Please provide an array containing EDA data and sampling rating")
# 	exit(0)

# eda = pd.array(sys.arv[0])
# sampling_rate = np.float(sys.arv[0])

eda_data = pd.DataFrame(pd.read_csv('/Users/halalotfy/project_us/case_eda_data.csv'))
eda = np.array((eda_data['eda'][0])[1:-1].split(',')).astype('float')
print(eda)
print(type(eda))
print(eda.shape)
eda = eda.astype('float')
print(eda)
# print(type(np.asarray(eda[0])))
# print(eda)
sampling_rate = 1000.0

aux, _, _ = signals.tools.filter_signal(signal=eda,
		                         ftype='butter',
		                         band='lowpass',
		                         order=4,
		                         frequency=5,
		                         sampling_rate=sampling_rate)
 
# smooth
sm_size = int(0.75 * sampling_rate)
filtered, _ = signals.tools.smoother(signal=aux,
                          kernel='boxzen',
                          size=sm_size,
                          mirror=True)


mean, med, mx, var, std_dev, abs_dev, kurt, skew = signals.tools.signal_stats(filtered)

try:
	onsets, peaks, amps = signals.eda.basic_scr(filtered, sampling_rate=sampling_rate)
except:
	onsets = np.array([])
	peaks = np.array([])
	amps = np.array([])

eda_features = pd.DataFrame(columns=['std_dev', 'max_diff', 'skewness', 'median_eda', 'num_peaks', 'mean_amp', 'max_amp', 'max_eda', 'min_eda', 'rise_time', 'kurt'])

# print({'eda_filtered':filtered, 'std_dev':std_dev, 'variance':var, 'max_diff':np.max(filtered) - np.min(filtered), 'skewness':skew, 'median_eda':med, 'mean_eda':mean, 'onsets': onsets, 'num_peaks':len(peaks), 'peaks':peaks, 'mean_amp':np.mean(amps), 'max_amp': mx, 'max_eda': np.max(filtered), 'min_eda': np.min(filtered), 'rise_time':np.mean((onsets-peaks)), 'kurt': kurt})

eda_features = eda_features.append({'std_dev':std_dev, 'max_diff':np.max(filtered) - np.min(filtered), 'skewness':skew, 'median_eda':med,'num_peaks':len(peaks), 'mean_amp':np.mean(amps), 'max_amp': mx, 'max_eda': np.max(filtered), 'min_eda': np.min(filtered), 'rise_time':np.mean((onsets-peaks)), 'kurt': kurt}, ignore_index=True)

scalar = joblib.load('./models/eda_scaler.pkl')
model = joblib.load('./models/nn_eda_model.pkl')
predict = model.predict(scalar.transform(eda_features))

print(predict)

