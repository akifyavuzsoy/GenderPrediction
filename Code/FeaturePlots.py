
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import random

from tqdm import tqdm
from FeatureSignals import Features

class featuresPlot:
    def __init__(self, main_path='../TIMIT_V2/data/' ,csv_train_file='../TIMIT_V2/train_data.csv', csv_test_file='../TIMIT_V2/test_data.csv', figure_save_path='../Datas/FeaturesImages/'):

        self.main_path = main_path
        self.csv_train_file = csv_train_file
        self.csv_test_file = csv_test_file
        self.figure_save_path = figure_save_path

        self.features = Features()

        self.df_train_data = None
        self.df_test_data = None
        self.y_train_labels = []
        self.y_test_labels = []
        self.wav_train_files = []
        self.wav_test_files = []

        self.x_testData = None
        self.x_trainData = None
        self.sr_testData = None
        self.sr_trainData = None

        self.spektrogram_test = None
        self.spektrogram_train = None
        self.mfcc_test = None
        self.mfcc_train = None
        self.zeroCrossingRate_test = None
        self.zeroCrossingRate_train = None
        self.spectralCentroid_test = None
        self.spectralCentroid_train = None
        self.spectralRolloff_test = None
        self.spectralRolloff_train = None
        self.chroma_test = None
        self.chroma_train = None
        self.spectralBant_test = None
        self.spectralBant_train = None

        self.random_count = random.randint(1, 100)
        self.figuresize_x = 15
        self.figuresize_y = 6

    def read_csv_files(self):
        print("Reading the CSV files...")
        self.df_train_data = pd.read_csv(self.csv_train_file)
        self.df_test_data = pd.read_csv(self.csv_test_file)
        print("Test File Count: ", self.df_test_data.shape)

    def get_Datas(self):
        self.read_csv_files()

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

        self.x_testData, self.sr_testData = librosa.load(self.wav_test_files[self.random_count], res_type='kaiser_fast')
        self.x_trainData, self.sr_trainData = librosa.load(self.wav_train_files[self.random_count], res_type='kaiser_fast')


    def extract_features(self):
        # --------------------------------------------------------------------------------------------------------------
        # Spektrogram Feature:
        self.spektrogram_test = self.features.spektrogram(self.x_testData, self.sr_testData)
        self.spektrogram_train = self.features.spektrogram(self.x_trainData, self.sr_trainData)
        print("*******************************************************************************************************")
        print("Spektrogram Feature Test Data:\n", self.spektrogram_test)
        print("Spektrogram Feature Train Data:\n", self.spektrogram_train)
        # For Test Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Spektrogram Test Data")
        librosa.display.specshow(self.spektrogram_test, sr=self.sr_testData, x_axis="time", y_axis="hz")
        plt.colorbar()
        plt.savefig(self.figure_save_path + "SpektrogramTestData")
        # For Train Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Spektrogram Train Data")
        librosa.display.specshow(self.spektrogram_train, sr=self.sr_trainData, x_axis="time", y_axis="hz")
        plt.colorbar()
        plt.savefig(self.figure_save_path + "SpektrogramTrainData")

        # --------------------------------------------------------------------------------------------------------------
        # MFCC Feature:
        self.mfcc_test = self.features.mfcc(self.x_testData, self.sr_testData, 40)
        self.mfcc_train = self.features.mfcc(self.x_trainData, self.sr_trainData, 40)
        print("*******************************************************************************************************")
        print("MFCC Feature Test Data:\n", self.mfcc_test)
        print("MFCC Feature Train Data:\n", self.mfcc_train)
        # For Test Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("MFCC Test Data")
        librosa.display.specshow(self.mfcc_test, x_axis="s")
        plt.colorbar()
        plt.savefig(self.figure_save_path + "mfccTestData")
        # For Train Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("MFCC Train Data")
        librosa.display.specshow(self.mfcc_train, x_axis="s")
        plt.colorbar()
        plt.savefig(self.figure_save_path + "mfccTrainData")

        # --------------------------------------------------------------------------------------------------------------
        # zeroCrossingRate Feature:
        self.zeroCrossingRate_test = self.features.zeroCrossingRate(self.x_testData)
        self.zeroCrossingRate_train = self.features.zeroCrossingRate(self.x_trainData)
        print("*******************************************************************************************************")
        print("zeroCrossingRate Feature Test Data:\n", self.zeroCrossingRate_test)
        print("zeroCrossingRate Feature Train Data:\n", self.zeroCrossingRate_test)

        # --------------------------------------------------------------------------------------------------------------
        # spectralCentroid Feature:
        self.spectralCentroid_test = self.features.spectralCentroid(self.x_testData)
        self.spectralCentroid_train = self.features.spectralCentroid(self.x_trainData)
        print("*******************************************************************************************************")
        print("spectralCentroid Feature Test Data:\n", self.spectralCentroid_test)
        print("spectralCentroid Feature Train Data:\n", self.spectralCentroid_train)
        # For Test Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Spectral Centroid Test Data")
        plt.semilogy(self.spectralCentroid_test.T)
        plt.ylabel("Hz")
        plt.savefig(self.figure_save_path + "spectralCentroidTest")
        # For Train Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Spectral Centroid Train Data")
        plt.semilogy(self.spectralCentroid_train.T)
        plt.ylabel("Hz")
        plt.savefig(self.figure_save_path + "spectralCentroidTrain")

        # --------------------------------------------------------------------------------------------------------------
        # spectralRolloff Feature:
        self.spectralRolloff_test = self.features.spectralRolloff(self.x_testData, self.sr_testData)
        self.spectralRolloff_train = self.features.spectralRolloff(self.x_trainData, self.sr_trainData)
        print("*******************************************************************************************************")
        print("spectralRolloff Feature Test Data:\n", self.spectralRolloff_test)
        print("spectralRolloff Feature Train Data:\n", self.spectralRolloff_train)
        # For Test Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Spectral Rolloff Test Data")
        plt.semilogy(self.spectralRolloff_test.T, "r")
        plt.ylabel("Hz")
        plt.savefig(self.figure_save_path + "spectralRolloffTest")
        # For Train Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Spectral Rolloff Train Data")
        plt.semilogy(self.spectralRolloff_train.T, "r")
        plt.ylabel("Hz")
        plt.savefig(self.figure_save_path + "spectralRolloffTrain")

        # --------------------------------------------------------------------------------------------------------------
        # chroma Feature:
        self.chroma_test = self.features.chroma(self.x_testData, self.sr_testData)
        self.chroma_train = self.features.chroma(self.x_trainData, self.sr_trainData)
        print("*******************************************************************************************************")
        print("chroma Feature Test Data:\n", self.chroma_test)
        print("chroma Feature Train Data:\n", self.chroma_train)
        # For Test Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Chroma Frekansı Test Data")
        librosa.display.specshow(self.chroma_test, y_axis="chroma", x_axis="time")
        plt.colorbar()
        plt.savefig(self.figure_save_path + "chromaTest")

        # For Train Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Chroma Frekansı Train Data")
        librosa.display.specshow(self.chroma_train, y_axis="chroma", x_axis="time")
        plt.colorbar()
        plt.savefig(self.figure_save_path + "chromaTrain")

        # --------------------------------------------------------------------------------------------------------------
        # spectralBant Feature:
        self.spectralBant_test = self.features.spectralBant(self.x_testData, self.sr_testData)
        self.spectralBant_train = self.features.spectralBant(self.x_trainData, self.sr_trainData)
        print("*******************************************************************************************************")
        print("spectralBant Feature Test Data:\n", self.spectralBant_test)
        print("spectralBant Feature Train Data:\n", self.spectralBant_train)
        # For Test Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Spektral Bant Genişliği Test Data")
        plt.semilogy(self.spectralBant_test.T, color='green')
        plt.ylabel("Hz")
        plt.xlabel("time(s)")
        plt.savefig(self.figure_save_path + "spectralBantTest")

        # For Train Data
        plt.figure(figsize=(self.figuresize_x, self.figuresize_y))
        plt.title("Spektral Bant Genişliği Train Data")
        plt.semilogy(self.spectralBant_train.T, color='green')
        plt.ylabel("Hz")
        plt.xlabel("time(s)")
        plt.savefig(self.figure_save_path + "spectralBantTrain")

        # --------------------------------------------------------------------------------------------------------------

    def showSignals(self):
        plt.figure(figsize=(10, 2))
        plt.title("Test Data")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        librosa.display.waveshow(self.x_testData, sr=self.sr_testData)
        plt.savefig(self.figure_save_path + "Test Data")
        #plt.show()

        plt.figure(figsize=(10, 2))
        plt.title("Train Data")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        librosa.display.waveshow(self.x_trainData, sr=self.sr_trainData)
        plt.savefig(self.figure_save_path + "Train Data")




if __name__ == "__main__":
    # Instantiate the class and run the process
    f_plot = featuresPlot()

    f_plot.get_Datas()
    f_plot.showSignals()
    f_plot.extract_features()

