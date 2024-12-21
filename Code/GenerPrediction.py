#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:33:56 2024

@author: akifyavuzsoy
"""

print("***********************************************************************")
print("             Started the Signal Classification Project")
print("***********************************************************************")

print("-----------------------------------------------------------------------")
print("             Libraries added successfully!")
print("-----------------------------------------------------------------------")

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, SimpleRNN, GRU, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from scipy.io import wavfile as wav

from FeatureSignals import Features

print(f"TensorFlow Version: {tf.__version__}")
print(f"Librosa Version: {librosa.__version__}")

class genderPrediction:
    def __init__(self, main_path, csv_train_file, csv_test_file, model_type, epochs=10, batch_size=32, model_save_path='Models/', model_path = None):
        super().__init__()
        
        self.main_path = main_path
        self.csv_train_file = csv_train_file
        self.csv_test_file = csv_test_file
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.model_path = model_path
        
        self.features = Features()  
        
        # Initialize arrays and variables
        self.y_train_labels = []
        self.y_test_labels = []
        self.wav_train_files = []
        self.wav_test_files = []
        self.x_train_array = []
        self.x_test_array = []
        self.sr_train_array = []
        self.sr_test_array = []
        self.mfcc_array_train = []
        self.mfcc_array_test = []
        
        self.rand_data = random.randint(1, 100)
        
        self.model = None
        
        self.score = 0
        
        # Create model save path if not exists
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def set_ModelType(self, model_type):
        self.model_type = model_type

    def set_Epochs(self, epochs):
        self.epochs = epochs

    def set_ModelPath(self, model_path):
        self.model_path = model_path

    def set_ModelSavePath(self, model_save_path):
        self.model_save_path = model_save_path
            
    def read_csv_files(self):
        print("Reading the CSV files...")
        self.df_train_data = pd.read_csv(self.csv_train_file)
        self.df_test_data = pd.read_csv(self.csv_test_file)
        
    def get_labels_and_files(self):
        print("Extracting Labels and WAV Files for Train and Test Datasets...")
        for index, row in tqdm(self.df_train_data.dropna(subset=['path_from_data_dir', 'speaker_id']).iterrows()):
            file_train = row['path_from_data_dir']
            if file_train.endswith('.WAV'):
                self.wav_train_files.append(self.main_path + file_train)
                self.y_train_labels.append(1 if row['speaker_id'][0] == 'M' else 0)
        
        for index, row in tqdm(self.df_test_data.dropna(subset=['path_from_data_dir', 'speaker_id']).iterrows()):
            file_test = row['path_from_data_dir']
            if file_test.endswith('.WAV'):
                self.wav_test_files.append(self.main_path + file_test)
                self.y_test_labels.append(1 if row['speaker_id'][0] == 'M' else 0)

    def load_wav_files(self, sample_rate = 16000, max_duration=3.0):
        print("Loading WAV files using Librosa...")
        max_samples = int(sample_rate * max_duration)
        for wav in tqdm(self.wav_train_files, desc="Train Files"):
            signal, sr = librosa.load(wav, res_type='kaiser_fast', sr=sample_rate)
            if len(signal) < max_samples:
                pad_width = max_samples - len(signal)
                signal = np.pad(signal, (0, pad_width), mode='constant')
            else:
                signal = signal[:max_samples]
            # All signals are ensured to be the same size...

            self.x_train_array.append(signal)
            self.sr_train_array.append(sr)
        
        for wav in tqdm(self.wav_test_files, desc="Test Files"):
            signal, sr = librosa.load(wav, res_type='kaiser_fast', sr=sample_rate)
            if len(signal) < max_samples:
                pad_width = max_samples - len(signal)
                signal = np.pad(signal, (0, pad_width), mode='constant')
            else:
                signal = signal[:max_samples]
            # All signals are ensured to be the same size...

            self.x_test_array.append(signal)
            self.sr_test_array.append(sr)
            
    def visualize_waveform(self, data_array, title="Waveform"):
        print(f"Visualizing {title}...")
        plt.figure(figsize=(12, 4))
        plt.plot(data_array)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        #plt.show()
        
    def read_Dateset(self):
        print("-----------------------------------------------------------------------")
        print("                 Reading the TIMIT Datasets")
        print("-----------------------------------------------------------------------")
        
        self.read_csv_files()
        self.get_labels_and_files()
        self.load_wav_files()
        self.visualize_waveform(self.x_test_array[self.rand_data])
        self.visualize_waveform(self.x_train_array[self.rand_data])
        
    def extract_features(self):
        print("Extracting MFCC Features...")
        for i in tqdm(range(len(self.x_train_array)), desc="Train MFCC"):
            mfcc_train = self.features.mfcc(self.x_train_array[i], self.sr_train_array[i], n_mfcc=40)
            self.mfcc_array_train.append(np.mean(mfcc_train.T, axis=0))

        for i in tqdm(range(len(self.x_test_array)), desc="Test MFCC"):
            mfcc_test = self.features.mfcc(self.x_test_array[i], self.sr_test_array[i], n_mfcc=40)
            self.mfcc_array_test.append(np.mean(mfcc_test.T, axis=0))
        print("Test data okundu...")
        # Convert to numpy arrays
        self.mfcc_array_train = np.array(self.mfcc_array_train)
        self.mfcc_array_test = np.array(self.mfcc_array_test)
        print("\nTest Shape: ", self.mfcc_array_test.shape)
        print("Train Shape: ", self.mfcc_array_train.shape)
        
    def build_model(self):
        print("-----------------------------------------------------------------------")
        print("                     Creating Model")
        print("-----------------------------------------------------------------------")
        
        match self.model_type:
            case 'CNN':
                print("Building the CNN Model...")
                self.model = None
                self.model = Sequential([
                    Dense(125, input_shape=(40,), activation='relu'),
                    Dropout(0.5),
                    Dense(250, activation='relu'),
                    Dropout(0.5),
                    Dense(125, activation='relu'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')  # For binary classification ('M' or 'F')
                ])
                
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                print("Model Summary: ")
                print(self.model.summary())
                
            case 'RNN':
                print("Building the RNN Model...")
                self.model = None
                self.model = Sequential([
                    SimpleRNN(128, input_shape=(40,1), activation='tanh'),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')
                ])
                
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                print("Model Summary:")
                self.model.summary()
            
            case 'LSTM':
                print("Building the LSTM Model...")
                self.model = None
                self.model = Sequential([
                    LSTM(128, input_shape=(40,1), return_sequences=True, activation='tanh'),
                    Dropout(0.5),
                    LSTM(64, return_sequences=False, activation='tanh'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')
                ])
                
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                print("LSTM Model Summary:")
                self.model.summary()

            case 'CoLSTM':
                print("Building the LSTM Model...")
                self.model = None
                self.model = Sequential([
                    # Conv1D Katmanı - Spektral özellikler için
                    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(40, 1)),
                    MaxPooling1D(pool_size=2),
                    Dropout(0.2),

                    # Conv1D Katmanı - Spektral özellikler için
                    Conv1D(filters=64, kernel_size=3, activation='relu'),
                    MaxPooling1D(pool_size=2),
                    Dropout(0.2),

                    # LSTM Katmanları - Zamansal ilişkileri öğrenmek için
                    LSTM(128, return_sequences=True, activation='relu'),
                    Dropout(0.5),
                    LSTM(64, return_sequences=False, activation='relu'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')
                ])

                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                print("LSTM Model Summary:")
                self.model.summary()

            case 'CNN_LSTM':
                print("Building the LSTM Model for CNN (feature extraction)...")
                sample_rate = 16000  # Ses örnekleme frekansı
                max_duration = 3.0  # Maksimum süre (saniye)
                input_shape = (int(sample_rate * max_duration), 1)  # Her sinyalin sabit boyutlu şekli

                self.model = None
                self.model = Sequential()

                # CNN Katmanları (1D Conv çünkü sinyal 1 boyutlu)
                self.model.add(Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape))
                self.model.add(MaxPooling1D(pool_size=2))
                self.model.add(BatchNormalization())

                self.model.add(Conv1D(32, kernel_size=5, activation='relu'))
                self.model.add(MaxPooling1D(pool_size=2))
                self.model.add(BatchNormalization())

                self.model.add(Conv1D(64, kernel_size=5, activation='relu'))
                self.model.add(MaxPooling1D(pool_size=2))
                self.model.add(BatchNormalization())

                self.model.add(Flatten())  # CNN'den çıkan öznitelikleri düzleştir

                # LSTM Katmanı
                self.model.add(Dense(128, activation='relu'))
                self.model.add(Dropout(0.5))
                self.model.add(Dense(64, activation='relu'))
                self.model.add(Dropout(0.5))

                # Çıkış Katmanı
                self.model.add(Dense(2, activation='softmax'))

                # Modeli derle
                self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


            case 'GRU':
                print("Building the GRU Model...")
                self.model = None
                self.model = Sequential([
                    GRU(128, input_shape=(40,1), return_sequences=True, activation='tanh'),
                    Dropout(0.5),
                    GRU(64, return_sequences=False, activation='tanh'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')  # İkili sınıflandırma için ('M' veya 'F')
                ])
                
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                print("GRU Model Summary:")
                self.model.summary()

            case _:
                pass
            
    def train_model(self):
        print("-----------------------------------------------------------------------")
        print("                     Training Model")
        print("-----------------------------------------------------------------------")
        
        model_name = f"{self.model_type}_{self.epochs}_best_model.keras"
        checkpoint = ModelCheckpoint(
            filepath=self.model_save_path + model_name,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        if (self.model_type == "CNN_LSTM"):
            # Since CNN will be used for feature extraction, raw signal data was given to the model...
            print("Selected CNN_LSTM Model...")
            x_train = np.array(self.x_train_array)
            x_test = np.array(self.x_test_array)
            x_train= x_train[..., np.newaxis]
            x_test = x_test[..., np.newaxis]

            y_train = to_categorical(self.y_train_labels, num_classes=2)
            y_test = to_categorical(self.y_test_labels, num_classes=2)

            history = self.model.fit(
                x_train,
                np.array(y_train),
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(x_test, np.array(y_test)),
                callbacks=[checkpoint],
                verbose=1
            )

            print("Evaluating the Model...")
            self.score = self.model.evaluate(x_test, np.array(y_test), verbose=0) # TODO: Burayı değiştir..
            print(f"Validation Accuracy: {self.score[1]}")

            predictions = self.model.predict(x_test) # TODO: Burayı değiştir..
            print("Predicted Test Dataset = ", predictions)

        else:
            history = self.model.fit(
                self.mfcc_array_train,
                np.array(self.y_train_labels),
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(self.mfcc_array_test, np.array(self.y_test_labels)),
                callbacks=[checkpoint],
                verbose=1
            )

            print("Evaluating the Model...")
            self.score = self.model.evaluate(self.mfcc_array_test, np.array(self.y_test_labels), verbose=0)
            print(f"Validation Accuracy: {self.score[1]}")

            predictions = self.model.predict(self.mfcc_array_test)
            predicted_classes = (predictions >= 0.5).astype(int).flatten()
            print("Predicted Test Dataset = ", predicted_classes)


        
        print("Saving the Model...")
        model_name = f"{self.model_type}_{self.epochs}_{self.score[1]}.h5"
        self.model.save(self.model_save_path + model_name)
        # TODO: model_save_path model tipine göre yeni klasör yapısı oluştur
        
        print("Plotting Training History...")
        model_accuracy_name = f"{self.model_type}_{self.epochs}_model_accuracy"
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.model_save_path + model_accuracy_name)
        #plt.show()

        model_loss_name = f"{self.model_type}_{self.epochs}_model_loss"
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.model_save_path + model_loss_name)
        #plt.show()
        
        return history, self.score[1]
    
    def test_model(self, audio_path):
        print("-----------------------------------------------------------------------")
        print("                     Testing Model")
        print("-----------------------------------------------------------------------")
        
        test_audio_data, test_sample_rate = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
        
        print(f"Time Series Length: {len(test_audio_data)}")
        print(f"Max Amplitude: {max(test_audio_data)}")
        print(f"Min Amplitude: {min(test_audio_data)}")
        
        test_mfcc = self.features.mfcc(test_audio_data, test_sample_rate, n_mfcc=40)
        print(f"MFCC Shape: {test_mfcc.shape}")
        
        test_mfcc_mean = np.mean(test_mfcc, axis=1).reshape(1,-1)
        test_prediction = self.model.predict(test_mfcc_mean)
        print("Predict Value: ", test_prediction)
        
        test_prediction_class = (test_prediction >= 0.5).astype(int)[0][0]
        test_prediction_label = "Male" if test_prediction_class == 1 else "Female"
        print(test_prediction_label)

        self.plot_audio_waveform(test_audio_data, test_sample_rate, test_prediction_label, audio_path, self.epochs, self.model_type)
        
        return test_prediction_class

    def plot_audio_waveform(self, y, sr, predicted_label, audio_path, epoch, model):
        # Sesin zaman serisini çizme
        plt.clf()  # Önceki grafikleri temizle
        librosa.display.waveshow(y, sr=sr)
        plt.title('Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # Ses dosyasının adını (uzantısız) al
        base_name = os.path.basename(audio_path)  # 'deneme1.wav'
        file_name, _ = os.path.splitext(base_name)  # 'deneme1'

        # Dosyanın klasör yolunu al
        folder_path = os.path.dirname(audio_path)  # Örneğin: 'C:/Projects/QT/TIMIT_Gender_Prediction'
        # Tam kaydetme yolunu oluşturma
        save_filename = os.path.join(folder_path, f"{file_name}_{model}_{epoch}_{predicted_label}.png")
        plt.savefig(save_filename)
        #plt.show()
        # self.canvas.draw()
    
    def load_model(self):
        self.model = None
        self.model = load_model(self.model_path)
        
        
            
if __name__ == "__main__":
    # Instantiate the class and run the process
    g_Pred = genderPrediction(
        main_path='../TIMIT_V2/data/',
        csv_train_file='../TIMIT_V2/train_data.csv',
        csv_test_file='../TIMIT_V2/test_data.csv',
        model_type='LSTM',
        epochs=12,
        batch_size=32
    )
    
    g_Pred.read_Dateset()
    g_Pred.extract_features()
    g_Pred.build_model()
    history = g_Pred.train_model()
    test_prediction_class = g_Pred.test_model('../Datas/deneme1.wav')
    
    
    
    
            
