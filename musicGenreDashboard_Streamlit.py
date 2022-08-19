#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 19:21:22 2022

@author: gk
"""



#source for the package source files for librosa, use this link while wanting to deploy streamlit onto github:
#   https://discuss.streamlit.io/t/unable-to-load-librosa-while-deployment/15706


import warnings
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
    
# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# converting dataset into .wav
from os import path
from pydub import AudioSegment
import sys

# librosa .load dependacies
import librosa
from pathlib import Path
#import ffmpeg

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title='Genre Classification Dashboard')


# Loading the available data and overview
path = "PinkPanther30.wav"
st.sidebar.header('Genre Classification')
st.sidebar.write('''This a Music Genre Classification App, that tries to predict which genre a music file belongs to. ''')
data = st.sidebar.file_uploader("Upload Dataset", type=['wav', 'au', 'mp3'])

# source for this code is at: https://blog.jcharistech.com/2021/01/21/how-to-save-uploaded-files-to-directory-in-streamlit-apps/



#Check for uploaded dataset
if data is not None:
    st.header('Play uploaded Dataset')
    st.audio(data.read(), format='audio/wav,au,mp3') 
       
    
# Default Dataset if none is uploaded
else:
    mssg = st.write(" This is an app for music genre classification. No music file has been uploaded. **Please upload a file to continue**.")
    exit(mssg)
    #st.header('Play default Dataset (*since none is uploaded*)')
    #st.audio(path, format='audio/wav')

	
#saving uploaded audio file
if data is not None:
    
    with open(os.path.join("test2",data.name),"wb") as f:
			  	f.write((data).getbuffer())
else:
	path



    
# Feature extraction of the Dataset
st.header('Feature Extraction of the Dataset')
# 1. Plotting the Signal
if data is not None:
    st.subheader('Plotting the Signal of Uploaded Dataset')
    src = (os.path.join("test2/",data.name))
    x, sr = librosa.load(src)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)
    st.pyplot(plt)


else:
    st.subheader('Plotting Signal of the default Dataset')
    x, sr = librosa.load(path)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)
    st.pyplot(plt)
    
#Zero Crossing Rate (ZCR)
if data is not None:
    st.subheader('ZCR of Uploaded Datatset')
    n0 = 90000
    n1 = 91000
    plt.figure(figsize=(14,5))
    plt.plot(x[n0:n1])
    plt.grid()
    st.pyplot(plt)
    
else:
    st.subheader('ZCR of Default Dataset')
    n0 = 90000
    n1 = 91000
    plt.figure(figsize=(14,5))
    plt.plot(x[n0:n1])
    plt.grid()
    st.pyplot(plt)

#MFCC
if data is not None:
    st.subheader('MFCC of Uploaded Dataset')
    x, fs = librosa.load(src)
    librosa.display.waveshow(x, sr=sr)
    st.pyplot(plt)
    
else:
    st.subheader('MFCC of default Dataset')
    x, fs = librosa.load(path)
    librosa.display.waveshow(x, sr=sr)
    st.pyplot(plt)

# Building the Model

df = pd.read_csv('audio.csv')

#preprocessing
class_list = df.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(class_list)

input_parameters = df.iloc[:, 1:27]
scaler = StandardScaler()
X = scaler.fit_transform(np.array(input_parameters))

#Training and validation tests
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.2)

# model
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation = 'relu', input_shape = (X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(45, activation = 'softmax'),
])


# Model training and evaluation
def trainModel(model, epochs, optimizer):
    #batch_size = 128
    batch_size = 256
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
    return model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = epochs, batch_size = batch_size)

#Launching the training, original epochs size was 100
model_history = trainModel(model = model, epochs = 100, optimizer = 'adam')

#Displaying Loss Curves
st.subheader('Loss')
loss_train_curve = model_history.history["loss"]
loss_val_curve = model_history.history["val_loss"]
plt.plot(loss_train_curve, label = "Train")
plt.plot(loss_val_curve, label = "Validation")
plt.legend(loc = 'upper right')
plt.title("Loss")
plt.show()
st.pyplot(plt)

#Displaying accuracy curves
st.subheader('Accuracy')
acc_train_curve = model_history.history["accuracy"]
acc_val_curve = model_history.history["val_accuracy"]
plt.plot(acc_train_curve, label = "Train")
plt.plot(acc_val_curve, label = "Validation")
plt.legend(loc = 'lower right')
plt.title("Accuracy")
plt.show()
st.pyplot(plt)

#Displaying accuracy of the model
st.subheader('Prediction Accuracy')
test_loss, test_acc = model.evaluate(X_val, y_val, batch_size = 128)
st.write("The test loss is: ", test_loss)
st.write("The best accuracy is: ", test_acc*100)


#Test data Preprocessing

#Defining Column names
header_test = "filename length chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean \
        spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var tempo mfcc1_mean mfcc1_var mfcc2_mean \
        mfcc2_var mfcc3_mean mfcc3_var mfcc4_mean mfcc4_var".split()
        

#Creating audio_test csv file
file = open('test2.csv', 'w', newline = '')
with file:
    writer = csv.writer(file)
    writer.writerow(header_test)
    
#Transform each .au file into .csv file
for filename in os.listdir(f"test2"):
    genre_name = f"test2/{filename}"
    #genre_name = f"audio_test2/Joeboy-Alcohol.wav"
    y, sr = librosa.load(genre_name, mono = True, duration = 90)
    chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
    rmse = librosa.feature.rms(y = y)
    spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
    spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
    rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y = y, sr = sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
        
    file = open('test2.csv', 'a', newline = '')
    
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
        
        

#Predictions
df_test = pd.read_csv('test2.csv')
X_test = scaler.transform(np.array(df_test.iloc[:, 1:27]))

# generate predictions for samples
predictions = model.predict(X_test)


# generate argmax for predictions
classes = np.argmax(predictions, axis = 1)

# transform classes number into classes name
result = encoder.inverse_transform(classes)
st.write('The music genre of the audio dataset you have uploaded is likely to be: ', df_test["filename"], result)
