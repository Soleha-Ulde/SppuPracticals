import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
# pip install tensorflow ignore if already installed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
dataset = pd.read_csv('C:/Users/Swift/OneDrive/Desktop/SppuSem7/SppuPracticals/Churn_Modelling.csv', index_col = 'RowNumber')
dataset.head()
# Features and target
X = dataset.iloc[:, 2:12]
X  # drop CustomerId and Surname
# Features and target
Y = dataset.iloc[:, 12].values  # Churn column
Y
pipeline = Pipeline([
    ('preprocess', ColumnTransformer(
        transformers=[
            ('gender', OneHotEncoder(drop='first'), ['Gender']),
            ('geo', OneHotEncoder(drop='first'), ['Geography'])
        ],
        remainder='passthrough'
    )),
    ('scaler', StandardScaler())
])
#Standardize the features
X = pipeline.fit_transform(X)
#Spilt the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# Build ANN
classifier = Sequential()
classifier.add(Dense(6, activation='relu', input_shape=(X_train.shape[1],)))
classifier.add(Dropout(0.1))
classifier.add(Dense(6, activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(1, activation='sigmoid'))
# Train ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = classifier.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=2)
y_pred = (classifier.predict(X_test) > 0.5).astype(int)
y_pred = classifier.predict(X_test)
print(y_pred[:5])
#Let us use confusion matrix with cutoff value as 0.5
y_pred = (y_pred > 0.5).astype(int)
print(y_pred[:5])
#Making the Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#Accuracy of our NN
print(((cm[0][0] + cm[1][1])* 100) / len(y_test), '% of data was classified correctly')