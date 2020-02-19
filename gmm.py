import os
from sklearn.mixture import GaussianMixture
from python_speech_features.base import mfcc
import numpy as np
import math
import scipy
from scipy.io.wavfile import read
import librosa
from sklearn.metrics import confusion_matrix

PATH='/home/sounak/Desktop/Semester_8/EE_624_Project/Project_2_data/'
wav_files=[f for f in os.listdir(PATH) if (f.endswith('.wav'))]
TEST_PATH='/home/sounak/Desktop/Semester_8/EE_624_Project/Project_2_data/Test_Data/'
test_wav_files=[f for f in os.listdir(TEST_PATH) if (f.endswith('.wav'))]

zero_data=np.empty(shape=(0,0))
one_data=np.empty(shape=(0,0))
two_data=np.empty(shape=(0,0))
three_data=np.empty(shape=(0,0))
four_data=np.empty(shape=(0,0))
five_data=np.empty(shape=(0,0))
six_data=np.empty(shape=(0,0))
seven_data=np.empty(shape=(0,0))
eight_data=np.empty(shape=(0,0))
nine_data=np.empty(shape=(0,0))


for file_name in wav_files:
    full_path=PATH+file_name
    class_name=file_name.split('_')[1]
    rate, data = read(full_path)
    mfcc_features = mfcc(data, samplerate=rate, winlen=0.01, winstep=0.003, numcep=13)
    delta1_mfcc = librosa.feature.delta(mfcc_features, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc_features, order=2)
    features=np.concatenate((mfcc_features, delta1_mfcc, delta2_mfcc), axis=1) 
    if class_name=='0':
        if zero_data.shape[0]==0:
            zero_data=features
        else:
            zero_data=np.append(zero_data, features, axis=0)
    if class_name=='1':
        if one_data.shape[0]==0:
            one_data=features
        else:
            one_data=np.append(one_data, features, axis=0)
    if class_name=='2':
        if two_data.shape[0]==0:
            two_data=features
        else:
            two_data=np.append(two_data, features, axis=0)
    if class_name=='3':
        if three_data.shape[0]==0:
            three_data=features
        else:
            three_data=np.append(three_data, features, axis=0)
    if class_name=='4':
        if four_data.shape[0]==0:
            four_data=features
        else:
            four_data=np.append(four_data, features, axis=0)
    if class_name=='5':
        if five_data.shape[0]==0:
            five_data=features
        else:
            five_data=np.append(five_data, features, axis=0)
    if class_name=='6':
        if six_data.shape[0]==0:
            six_data=features
        else:
            six_data=np.append(six_data, features, axis=0)
    if class_name=='7':
        if seven_data.shape[0]==0:
            seven_data=features
        else:
            seven_data=np.append(seven_data, features, axis=0)
    if class_name=='8':
        if eight_data.shape[0]==0:
            eight_data=features
        else:
            eight_data=np.append(eight_data, features, axis=0)
    if class_name=='9':
        if nine_data.shape[0]==0:
            nine_data=features
        else:
            nine_data=np.append(nine_data, features, axis=0)    
            
print(zero_data.shape)
print(one_data.shape)
print(two_data.shape)
print(three_data.shape)
print(four_data.shape)
print(five_data.shape)
print(six_data.shape)
print(seven_data.shape)
print(eight_data.shape)
print(nine_data.shape)

 
gmm_0=GaussianMixture(n_components=32)
gmm_1=GaussianMixture(n_components=32)
gmm_2=GaussianMixture(n_components=32)
gmm_3=GaussianMixture(n_components=32)
gmm_4=GaussianMixture(n_components=32)
gmm_5=GaussianMixture(n_components=32)
gmm_6=GaussianMixture(n_components=32)
gmm_7=GaussianMixture(n_components=32)
gmm_8=GaussianMixture(n_components=32)
gmm_9=GaussianMixture(n_components=32)
    
gmm_0.fit(zero_data)
gmm_1.fit(one_data)
gmm_2.fit(two_data)
gmm_3.fit(three_data)
gmm_4.fit(four_data)
gmm_5.fit(five_data)
gmm_6.fit(six_data)
gmm_7.fit(seven_data)
gmm_8.fit(eight_data)
gmm_9.fit(nine_data)

y_pred=[]
y_true=[]
for test_file in test_wav_files:
    full_test_path=TEST_PATH+test_file
    class_name=test_file.split('_')[1]
    rate, data = read(full_test_path)
    mfcc_features = mfcc(data, samplerate=rate, winlen=0.01, winstep=0.003, numcep=13)
    delta1_mfcc = librosa.feature.delta(mfcc_features, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc_features, order=2)
    features=np.concatenate((mfcc_features, delta1_mfcc, delta2_mfcc), axis=1)
    probabilities=np.zeros((10))
    probabilities[0]=np.sum(gmm_0.score_samples(features))
    probabilities[1]=np.sum(gmm_1.score_samples(features))
    probabilities[2]=np.sum(gmm_2.score_samples(features))
    probabilities[3]=np.sum(gmm_3.score_samples(features))
    probabilities[4]=np.sum(gmm_4.score_samples(features))
    probabilities[5]=np.sum(gmm_5.score_samples(features))
    probabilities[6]=np.sum(gmm_6.score_samples(features))
    probabilities[7]=np.sum(gmm_7.score_samples(features))
    probabilities[8]=np.sum(gmm_8.score_samples(features))
    probabilities[9]=np.sum(gmm_9.score_samples(features))
    y_pred.append(np.argmax(probabilities))
    y_true.append(int(class_name))
   
mat=confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(mat)
    
