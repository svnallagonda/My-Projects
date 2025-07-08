import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma: # stft needs to be calculated only if chroma is true
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            # For melspectrogram, 'y' (audio time series) is the correct argument, not 'X' directly if X is already an audio time series.
            # Assuming X is the audio time series, the original code might have passed it correctly, but let's be explicit.
            mel_spec=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel_spec))
    return result

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    # The original glob path was only looking for a single file.
    # To load the entire RAVDESS dataset, you need to glob over all subdirectories.
    # Assuming your RAVDESS dataset is structured like:
    # speech-emotion-recognition-ravdess-data/Actor_01/03-01-01-01-01-01-01.wav
    # You should adjust the path to match your actual dataset location.
    # A common structure for RAVDESS is 'Actor_xx' folders containing the audio files.
    # Let's assume the base directory containing Actor_xx folders is 'D:/SURYA/speech-emotion-recognition-ravdess-data/'.
    # We'll use a more generic glob pattern to pick up all WAV files.

    # IMPORTANT: Replace this with the actual path to your RAVDESS dataset.
    # For example, if your dataset is in 'D:\RAVDESS_Dataset\Actor_01\*.wav', you'd use that.
    # A common structure is a main folder containing 'Actor_01', 'Actor_02', etc.
    data_path = "D:\SURYA\speech-emotion-recognition-ravdess-data\Actor_01"
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)

                # Ensure the file_name has enough parts after splitting by '-'
                parts = file_name.split("-")
                if len(parts) < 3: # Check if the split results in at least 3 parts
                    print(f"Skipping malformed file name: {file_name}")
                    continue

                emotion_code = parts[2]
                if emotion_code not in emotions:
                    print(f"Skipping file with unknown emotion code: {file_name}")
                    continue

                emotion = emotions[emotion_code]

                if emotion not in observed_emotions:
                    continue

                feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#DataFlair - Get the shape of the training and testing datasets
print(f"Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}")

#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#DataFlair - Train the model
print("Training the model...")
model.fit(x_train,y_train)
print("Model training complete.")

#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


# for dataset: https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view?usp=drive_link
