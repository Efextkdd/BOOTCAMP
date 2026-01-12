# Import Data Analysis libraries
import pandas as pd
import numpy as np
import os # -> This library used for interacting with the operating system and its environment
#Import Data visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
import pickle
warnings.filterwarnings('ignore')
# working directory for the TESS dataset
Tess =  'C:/Users/Admin/Desktop/ProjectData/TESS3/TESS1/TESS Toronto emotional speech set data'    

paths = [] #assigns an empty array to paths variable
labels = []

#gets and joins the directory  and the audio files
for dirname,_,filenames in os.walk(Tess):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800: #length of paths
        print('Dataset is Loaded')

print(len(paths))
print(paths[:5])
print(labels[:5])

df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
print(df.head())

sns.countplot(data=df, x='label')
plt.show()

#Feature Extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
extract_mfcc(df['speech'][0])
X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X_mfcc

#Data preparation
X = [x for x in X_mfcc]
X = np.array(X)
print(X.shape)

pd.DataFrame(X).head()

#Label Encoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['label'])  # 1D array

#Random Forest Model
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
# Evaluate the model and calculate metrics
from sklearn.metrics import accuracy_score
rdtree = RandomForestClassifier(n_estimators=90)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rdtree.fit(X_train,y_train)
y_pred = rdtree.predict(X_test)
print(f'The accuracy score is: {accuracy_score(y_test,y_pred)*100:.2f}%')

# iterating when K is 1,2 to 40, each loop gets the average  error rate(misclassification)
error_rate_rf = []
best_error = 1
best_model = None

for a in range(1,100):
    model = RandomForestClassifier(n_estimators=a, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    error = np.mean(preds != y_test)
    error_rate_rf.append(error)

    if error < best_error:
        best_error = error
        best_model = model

#Check the K value with the lowest error rate, however the value with the lowest error rate is the best K value for quality prediction
#Check the K value with the lowest error rate, however the value with the lowest error rate is the best K value for quality prediction
sns.set_style('darkgrid')
plt.figure(figsize =(10,6))
plt.plot(range(1,100),error_rate_rf, linestyle = '--', marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error_rate vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Error_rate')
plt.show()

with open("emotion_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
    
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model trained and saved as emotion_model.pkl")