from data_parsing import main as get_dataset
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras.models import Model
from keras import optimizers
from keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay, confusion_matrix, classification_report
import numpy as np
import torch 
import tensorflow as tf
from matplotlib import pyplot as plt
import itertools

'''
model = Sequential()
    model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(22, 20), name='blstm1')) 
    model.add(Bidirectional(LSTM(32), name='blstm2'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(22, activation='softmax', name='fc2'))

'''

def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(1, 20), name='blstm1')) 
    model.add(Bidirectional(LSTM(32), name='blstm2'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(22, activation='softmax', name='fc2'))
    
    rmsprop = optimizers.RMSprop(lr=0.001)
        
    model.compile(loss="categorical_crossentropy", optimizer=rmsprop, metrics=['accuracy'])

    model.summary()

    return model

def main():
    nsplits = 5
    #cv = StratifiedShuffleSplit(n_splits=nsplits, train_size=0.8)
    scores = []
    model = create_model()
    X_train, X_test, y_train, y_test = get_dataset()

    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=500, batch_size=20, verbose=1)
    eval = model.evaluate(X_test, y_test)
    scores.append(eval)
    print ('Loss: ' + str(eval[0]) + ' ' + 'Acc: ' + str(eval[1]))
            
    prediction = model.predict(X_test)
    y_true = y_test.argmax(axis=-1)
    y_pred = prediction.argmax(axis=-1)

    #plot confusion
    cm = confusion_matrix(y_true, y_pred)
    cmap=plt.cm.Blues
    title = "CM"
    classes = ["a","b","c","d","e","f","h","i","k","l","m","n","o","p","q","r","t","u","v","w","x","y"]
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap) 
    plt.title(title) 
    plt.colorbar() 
    tick_marks = np.arange(len(classes)) 
    plt.xticks(tick_marks, classes, rotation=45) 
    plt.yticks(tick_marks, classes) 
 
    thresh = cm.max() / 2. 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, cm[i, j], 
                 horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black") 
 
    plt.tight_layout() 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label') 
    plt.show() 

main()