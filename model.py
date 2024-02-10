from data_parsing import main as get_dataset
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras.models import Model
from keras import optimizers
from keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay, confusion_matrix, classification_report, auc
import numpy as np
import torch 
import os
import tensorflow as tf
from tensorflow import lite
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

def load_callbacks():
    import os
    from keras.callbacks import EarlyStopping, History, ModelCheckpoint

    history = History()
    checkpoint_path = "checkpoints/chpk.h5"
    checkpoint_path_best_only = "bestcheckpoints/chpk.h5"


    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  save_best_only=False,
                                  monitor='val_accuracy',
                                  mode='max',
                                  verbose=1)
    cp_callback_best_only = ModelCheckpoint(filepath=checkpoint_path_best_only,
                                            save_weights_only=True,
                                            save_best_only=True,
                                            monitor='val_accuracy',
                                            mode='max',
                                            verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    return [history, cp_callback, cp_callback_best_only, early_stopping]

def main():
    nsplits = 5
    #cv = StratifiedShuffleSplit(n_splits=nsplits, train_size=0.8)
    scores = []
    model = create_model()
    X_train, X_test, y_train, y_test = get_dataset()

    callbacks = load_callbacks()

    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=500, batch_size=20, verbose=1, callbacks=callbacks)
    eval = model.evaluate(X_test, y_test)
    scores.append(eval)
    print ('Loss: ' + str(eval[0]) + ' ' + 'Acc: ' + str(eval[1]))

    model.save('model/mymodel.keras')
    converter = lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter=True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS]

    tfmodel = converter.convert()
    open('model/exportedmodel.task', 'wb').write(tfmodel)

main()