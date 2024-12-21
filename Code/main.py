from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import QTimer, pyqtSignal
import sys
import sounddevice as sd
from scipy.io.wavfile import write
import os
import re
from datetime import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from tensorflow.keras.models import load_model
from GenerPrediction import genderPrediction

from mainUI import Ui_MainWindow  # Orijinal dosyanızdaki sınıfı içe aktarıyoruz

maj_version = 1
min_version = 3
rev_version = 4

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()

        # UI kurulumunu yapıyoruz
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # MediaPlayer örneği
        self.player = QMediaPlayer()

        self.is_duplicate_model_F = False
        self.is_readData_F = False
        self.trainModel_accuracy = 0

        self.g_Pred = genderPrediction(
            main_path='../TIMIT_V2/data/',
            csv_train_file='../TIMIT_V2/train_data.csv',
            csv_test_file='../TIMIT_V2/test_data.csv',
            model_type=None,
            epochs=None,
            batch_size=32,
            model_save_path='../Models/',
            model_path=None
        )
        if (maj_version < 10):
            self.version = f"0{maj_version}"
        else:
            self.version = f"{maj_version}"
        if (min_version < 10):
            self.version += f".0{min_version}"
        else:
            self.version += f".{min_version}"
        if (rev_version < 10):
            self.version += f".0{rev_version}"
        else:
            self.version += f".{rev_version}"
        #self.version = f"V{maj_version}.{min_version}.{rev_version}"
        print(f"Program Versiyonu: {self.version}")
        self.ui.lbl_version.setText("Ver " + self.version)

        # Open Audio butonuna tıklama olayını bağla
        self.ui.btn_OpenAudio.clicked.connect(self.play_audio)

        # Train Model butonuna tıklanıldığında tetiklenecek fonksiyonu bağlıyoruz
        self.ui.btn_TrainModel.clicked.connect(self.train_model)

        # Yeni Model Path Seçme butonuna tıklanıldığında klasör seçimi yapacak fonksiyon
        self.ui.tBtn_NewSelectModel.clicked.connect(self.select_new_trainModel_path)

        # Test Model butonuna tıklanıldığında tetiklenecek fonksiyonu bağlıyoruz
        self.ui.btn_TestModel.clicked.connect(self.test_model)

        # Model Seçme butonuna tıklanıldığında dosya seçimi yapacak fonksiyon
        self.ui.tBtn_SelectModel.clicked.connect(self.select_testModel_file)

        # Ses Dosyası Seçme butonuna tıklanıldığında dosya seçimi yapacak fonksiyon
        self.ui.tBtn_SelectFile.clicked.connect(self.select_audio_file)

        # Yeni Ses Kaydı Yap butonuna tıklanıldığında ses kaydı yapacak fonksiyon
        self.ui.btn_NewAudio.clicked.connect(self.record_audio)

    def train_model(self):
        # cBox_TrainModel (ComboBox) ve sBox_TrainModel (SpinBox) değerlerini alıyoruz
        selected_model = self.ui.cBox_TrainModel.currentText()
        epoch_count = self.ui.sBox_TrainModel.value()
        model_save_path = self.ui.txt_NewModelPath.text()

        if model_save_path:
            model_save_path = model_save_path + '/' + selected_model + '/'
            self.g_Pred.set_ModelSavePath(model_save_path)
        else:
            model_save_path = '../Models/' + selected_model + '/'
            self.g_Pred.set_ModelSavePath(model_save_path)

        if epoch_count != 0 and selected_model:
            for file_name in os.listdir(model_save_path):
                if file_name.endswith('.h5'):
                    parts = file_name.split('_')
                    existing_model_type = parts[0]
                    existing_epoch = parts[1]
                    if existing_model_type == selected_model and existing_epoch == str(epoch_count):
                        self.is_duplicate_model_F = True
                        accuracy_str = parts[2].rsplit('.', 1)[0]
                        self.trainModel_accuracy = round(float(accuracy_str), 2)
                        break  # Aynı model türü ve epoch değeri bulundu
                else:
                    continue

            if self.is_duplicate_model_F:
                pass
            else:
                self.g_Pred.set_ModelType(selected_model)
                self.g_Pred.set_Epochs(epoch_count)

                # Konsola yazdırıyoruz (veya başka işlemler için kullanabilirsiniz)
                print(f"Selected Model: {selected_model}")
                print(f"Epoch Count: {epoch_count}")

                if not self.is_readData_F:
                    self.g_Pred.read_Dateset()
                    self.g_Pred.extract_features()
                    self.is_readData_F = True

                self.g_Pred.build_model()
                history, self.trainModel_accuracy = self.g_Pred.train_model()

            self.ui.txt_ModelAccuracy.setText(f"{self.trainModel_accuracy:.2f}")
            QtWidgets.QMessageBox.information(self, "Success", "Model Training Completed!")

            # TODO: Eğitim esnasında bir bar dolsun


        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select all parameters!!!")
            print("Please select all parameters!!!")

    def test_model(self):
        # Test model için gerekli alanları alıyoruz
        model_path = self.ui.txt_ModelPath.text()
        audio_path = self.ui.txt_AudioPath.text()

        # Eğer txt_AudioPath alanı boş değilse, wav dosyasını oku
        if model_path and audio_path and audio_path.endswith('.wav'):

            # Konsola yazdırıyoruz
            print(f"Model Path: {model_path}")
            print(f"Audio Path: {audio_path}")
            try:
                file_name = os.path.basename(model_path)
                parts = file_name.split('_')
                model_type = parts[0]
                epochs = parts[1]
                self.g_Pred.set_ModelType(model_type)
                self.g_Pred.set_Epochs(epochs)
                self.g_Pred.set_ModelPath(model_path)
                self.g_Pred.load_model()
                predicted_class = self.g_Pred.test_model(audio_path)
                predicted_label = "Male" if predicted_class == 1 else "Female"
                print("Predicted Class: ", predicted_label)

                # Regex deseni: CNN_epoch_accuracy.h5
                pattern = r'(CNN|DNN|RNN|LSTM|GRU)_(\d+)_(\d\.\d+)\.h5'

                # Dosya adını dosya yolundan ayırarak regex ile eşleşme kontrolü yapabiliriz.
                match = re.search(pattern, model_path)
                if match:
                    epoch = int(match.group(2))  # Epoch sayısı
                    accuracy = match.group(3)  # Accuracy değeri
                    accuracy = "{:.2f}".format(float(accuracy))
                    print(f"Epoch: {epoch}")
                    print(f"Accuracy: {accuracy}")
                    self.ui.txt_TestModelAccuracy.setText(accuracy)

                else:
                    print("Dosya adı desene uymuyor!")

                self.ui.txt_Prediction.setText(predicted_label)

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error reading audio file: {e}")
                print(f"Error reading audio file: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select all parameters!!!")
            print("Please select all parameters!!!")

    def select_new_trainModel_path(self):
        # Yeni bir klasör seçimi yapıyoruz
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder_path:
            # Seçilen klasör yolunu txt_NewModelPath alanına yazdırıyoruz
            self.ui.txt_NewModelPath.setText(folder_path)
            print(f"New Model Path: {folder_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a file")
            print("Please select a file")

    def select_testModel_file(self):
        # Model dosyası seçimi için dialog açıyoruz
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Model File", "",
                                                   "Select Model (*.h5 );;All Files (*.*)")

        if file_path:
            self.ui.txt_ModelPath.setText(file_path)
            print(f"Selected Model File: {file_path}")
            return True
        else:
            QtWidgets.QMessageBox.warning(None, "Warning", "Please select a file")
            print("Please select a file")
            return False


    def select_audio_file(self):
        # Ses dosyası seçimi için dialog açıyoruz
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "",
                                                   "Audio Files (*.wav *.mp3);;All Files (*)")
        if file_path:
            self.ui.txt_AudioPath.setText(file_path)
            print(f"Selected Audio File: {file_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a file")
            print("Please select a file")

    def record_audio(self):
        # Ses kayıt ayarları
        sample_rate = 44100  # Örnekleme hızı (Hz)
        duration = 5  # Kayıt süresi (saniye)

        # Kullanıcıdan dosya adı almak için
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Audio File", "", "WAV Files (*.wav);;All Files (*)",
                                                   options=options)

        if file_path:
            # Mikrofon ile ses kaydı al
            print("Recording started...")
            # TODO: Ses kaydeidldiğini kullanıcıya da göster
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
            sd.wait()  # Kayıt tamamlanana kadar bekle
            print("Recording finished.")

            # Kaydedilen sesi wav dosyasına yaz
            write(file_path, sample_rate, recording)
            print(f"Audio saved as: {file_path}")

            # Kaydedilen dosya yolunu txt_AudioPath alanına yaz
            self.ui.txt_AudioPath.setText(file_path)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a file")
            print("Please select a file")

    def play_audio(self):
        # txt_AudioPath alanındaki .wav dosyasını al
        audio_path = self.ui.txt_AudioPath.text().strip()

        # Dosya yolunun boş olup olmadığını kontrol et
        if not audio_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select an audio file.")
            return

        # Dosyanın .wav uzantısına sahip olup olmadığını kontrol et
        if not audio_path.endswith('.wav'):
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a valid .wav file.")
            return

        try:
            # QMediaPlayer kullanarak ses dosyasını oynat
            url = QUrl.fromLocalFile(audio_path)
            content = QMediaContent(url)
            self.player.setMedia(content)
            self.player.setVolume(50)  # Ses yüksekliği (0-100 arası)
            self.player.play()
            print(f"Playing audio: {audio_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error playing audio file: {e}")
            print(f"Error playing audio file: {e}")

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
        plt.show()
        #self.canvas.draw()



if __name__ == "__main__":
    # Uygulama nesnesi oluşturulur
    app = QtWidgets.QApplication(sys.argv)

    # Ana pencere oluşturulur ve gösterilir
    window = MainApp()
    window.show()

    # Uygulamanın çalıştırılması
    sys.exit(app.exec_())
