import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, BatchNormalization

# Veri seti yolunu belirtin
dataset_path = "TIMIT_PATH"  # TIMIT veri setinizin kök dizini

# Ses dosyalarını ve etiketleri yükleme
def load_timit_data(dataset_path, sample_rate=16000, max_duration=3.0):
    audio_signals = []
    labels = []
    max_samples = int(sample_rate * max_duration)  # Maksimum örnek sayısı
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Ses dosyasını yükleme ve normalizasyon
                signal, sr = librosa.load(file_path, sr=sample_rate)
                # Sinyali belirtilen uzunluğa padding veya truncation ile sabitleme
                if len(signal) < max_samples:
                    pad_width = max_samples - len(signal)
                    signal = np.pad(signal, (0, pad_width), mode='constant')
                else:
                    signal = signal[:max_samples]
                audio_signals.append(signal)
                # Dosya adından cinsiyeti çıkarın ('DR1_MKLS0.wav' gibi bir format bekleniyor)
                gender = "male" if file[4] == "M" else "female"
                labels.append(gender)
    return np.array(audio_signals), np.array(labels)

# Veri ve etiketleri yükle
sample_rate = 16000  # Ses örnekleme frekansı
max_duration = 3.0   # Maksimum süre (saniye)
audio_signals, labels = load_timit_data(dataset_path, sample_rate, max_duration)

# Etiketleri encode et
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = np.eye(2)[labels_encoded]  # One-hot encoding

# Eğitim ve test veri setini ayır
X_train, X_test, y_train, y_test = train_test_split(audio_signals, labels_categorical, test_size=0.2, random_state=42)

# CNN + LSTM Modeli
input_shape = (int(sample_rate * max_duration), 1)  # Her sinyalin sabit boyutlu şekli

model = Sequential()

# CNN Katmanları (1D Conv çünkü sinyal 1 boyutlu)
model.add(Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv1D(32, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv1D(64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Flatten())  # CNN'den çıkan öznitelikleri düzleştir

# LSTM Katmanı
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Çıkış Katmanı
model.add(Dense(2, activation='softmax'))

# Modeli derle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
X_train = X_train[..., np.newaxis]  # Kanal boyutunu ekleyin (1D Conv için)
X_test = X_test[..., np.newaxis]

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Test setindeki başarıyı değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")