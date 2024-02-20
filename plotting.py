import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay, confusion_matrix, classification_report
import seaborn as sns
import itertools

#loads the wights of our model
def load_model(epoch):
    # Load the previously saved weights
    mymodel = create_model()
    if epoch >= 0:
        mymodel.load_weights(f'checkpoints/chpk.h5')
    else:
        mymodel.load_weights(f'bestcheckpoints/chpk.h5')

    return mymodel

#functions that calculate tp,tn,fp,tp
def true_positive(y_true, y_pred):
    tp = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 1 and yp == 1:
            tp += 1
    
    return tp

def true_negative(y_true, y_pred):

    tn = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 0 and yp == 0:
            tn += 1
            
    return tn

def false_positive(y_true, y_pred):

    fp = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 0 and yp == 1:
            fp += 1
            
    return fp

def false_negative(y_true, y_pred):

    fn = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 1 and yp == 0:
            fn += 1
            
    return fn

#show the metrics
def report(model, X_test, y_test, history=None):

    prediction = model.predict(X_test, verbose=0)
   
    #plot confusion
    y_true = y_test.argmax(axis=-1)
    y_pred = prediction.argmax(axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    cmap=plt.cm.Blues
    title = "CM"
    moving_classes = ["j","no","si","z"]
    classes = ["a","b","c","d","e","f","g","h","i","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]
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
    if history is not None:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    from model import create_model, create_moving_model
    from data_parsing import main

    # generate a never seen dataset
    X_train, X_test, y_train, y_test = main()

    # load model
    model = load_model(-1)
    report(model, X_train,y_train)
