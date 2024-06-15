import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
from keras.optimizers import Adam
import os
from IPython.display import display

def loadData(dataset,split=0):
    if(split == 0):
        print("In non split")
        return tf.keras.preprocessing.image_dataset_from_directory(f"../Data/{dataset}",
                                                                   labels='inferred',shuffle=True,seed=51)
    else:
        print("In split")
        return tf.keras.preprocessing.image_dataset_from_directory(f"../Data/{dataset}",
                                                                   labels='inferred',shuffle=True,seed=51,
                                                            validation_split=split,subset='both')
    

#from https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
def f1_score(y_true, y_pred): #taken from old keras source code 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def createModel(convFilters, convAndPoolLayers,numberOfFCLayers, numberOfNeuronsPerFCLayer, dropoutRate):
    model = keras.Sequential()
        
    for convAndPoolLayer in range(convAndPoolLayers):
        model.add(keras.layers.Conv2D(convFilters,(3,3), activation='relu',padding='valid'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='valid'))
        
    model.add(keras.layers.Flatten())
    
    for layer in range(numberOfFCLayers):
        if layer == numberOfFCLayers - 1:
            model.add(keras.layers.Dense(1,activation='softmax'))
        else:
            model.add(keras.layers.Dense(numberOfNeuronsPerFCLayer,activation='relu'))
            model.add(keras.layers.Dropout(dropoutRate))
             
    model.compile(optimizer="adam",loss='binary_crossentropy', metrics=f1_score)
    return model

def createModelParametersDF(n_convFilters,n_convAndPoolLayers,
                            n_FCLayers,n_NeuronsPerFCLayers,n_Epochs,dropoutRates,trainScores,devScores):
    modelParameters = dict()
    modelParameters['n_convFilters'] = n_convFilters
    modelParameters['n_convAndPoolLayers'] = n_convAndPoolLayers
    modelParameters['n_FCLayers'] = n_FCLayers
    modelParameters['n_NeuronsPerFCLayers'] = n_NeuronsPerFCLayers
    modelParameters['n_Epochs'] = n_Epochs
    modelParameters['dropoutRate'] = dropoutRates
    modelParameters['trainScore'] = trainScores
    modelParameters['devScore'] = devScores

    modelParametersDF = pd.DataFrame(modelParameters, columns=modelParameters.keys())
    return modelParametersDF

def createRangeFromMidpoint(midpoint,range,mandatoryMinimum=1):
    possibleMin = int(midpoint-(range/2))
    possibleMin = max([mandatoryMinimum,possibleMin])
    possibleMax = int(midpoint+(range/2))
    possibleRange = np.arange(possibleMin,possibleMax)
    return possibleRange