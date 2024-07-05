#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
from keras.optimizers import Adam
import os
from IPython.display import display
get_ipython().run_line_magic('autosave', '5')


# In[2]:


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


# In[3]:


def f1_score(y_true, y_pred): #taken from old keras source code 
    true_positives = K.sum(y_true*y_pred)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# In[4]:


def createModel(convFilters, convAndPoolLayers,numberOfFCLayers, numberOfNeuronsPerFCLayer, dropoutRate, adamLearningRate,L2Rate):
    model = keras.Sequential()
        
    for convAndPoolLayer in range(convAndPoolLayers):
        model.add(keras.layers.Conv2D(convFilters,(3,3), activation='relu',padding='valid'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='valid'))
        
    model.add(keras.layers.Flatten())
    
    for layer in range(numberOfFCLayers):
        if layer == numberOfFCLayers - 1:
            model.add(keras.layers.Dense(1,activation='softmax'))
        else:
            model.add(keras.layers.Dense(numberOfNeuronsPerFCLayer,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2Rate)))
            model.add(keras.layers.Dropout(dropoutRate))
            
    adamOptimizer = keras.optimizers.legacy.Adam(learning_rate=adamLearningRate)       
    model.compile(optimizer="adam",loss=f1_loss, metrics=f1_score)
    return model


# In[5]:


def createModelParametersDF(n_convFilters,n_convAndPoolLayers,
                            n_FCLayers,n_NeuronsPerFCLayers,n_Epochs,
                            dropoutRates,adamLearningRates,L2Rates,trainScores,devScores):
    modelParameters = dict()
    modelParameters['n_convFilters'] = n_convFilters
    modelParameters['n_convAndPoolLayers'] = n_convAndPoolLayers
    modelParameters['n_FCLayers'] = n_FCLayers
    modelParameters['n_NeuronsPerFCLayers'] = n_NeuronsPerFCLayers
    modelParameters['n_Epochs'] = n_Epochs
    modelParameters['dropoutRate'] = dropoutRates
    modelParameters['adamLearningRates'] = adamLearningRates
    modelParameters['L2Rates'] = L2Rates
    modelParameters['trainScore'] = trainScores
    modelParameters['devScore'] = devScores

    modelParametersDF = pd.DataFrame(modelParameters, columns=modelParameters.keys())
    return modelParametersDF


# In[6]:


def createRangeFromMidpoint(midpoint,range,mandatoryMinimum=1):
    possibleMin = int(midpoint-(range/2))
    possibleMin = max([mandatoryMinimum,possibleMin])
    possibleMax = int(midpoint+(range/2))
    possibleRange = np.arange(possibleMin,possibleMax)
    return possibleRange


# In[7]:


def f1_loss(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = (2 * precision * recall) / (precision + recall + K.epsilon())
    return 1 - f1


# In[8]:


def generateAdamLearningRate(lastBestLearningRate=None):
    learningRatePower = np.random.random() * (-2 + 4) - 4
    learningRate = np.power(10,learningRatePower)
    if lastBestLearningRate != None:
        learningRate = (learningRate + lastBestLearningRate) / 2
    return learningRate


# In[9]:


def generateL2(lastBestL2=None):
    l2Power = np.random.random() * (3 + 2) - 2
    l2 = np.power(10,l2Power)
    if lastBestL2 != None:
        l2 = (l2 + lastBestL2) / 2
    return l2


# In[10]:


def main():

    possibleConvFilters = createRangeFromMidpoint(32,32)

    possibleConvAndPoolLayers = createRangeFromMidpoint(3,6)

    possibleNumberOfFCLayers = createRangeFromMidpoint(16,16)
    possibleNumberOfNeuronsPerFCLayer = createRangeFromMidpoint(16,16)

    possibleNumberOfEpochs = createRangeFromMidpoint(20,20)
    trial = 0
    
    train = loadData("training")
    dev,test = loadData("testing",.5)

    choose = np.random.choice

    n_convFilters = []
    n_convAndPoolLayers = []
    n_FCLayers = []
    n_NeuronsPerFCLayers = []
    n_Epochs = []
    dropoutRates = []
    adamLearningRates = []
    L2Rates = []
    trainScores = []
    devScores = []
    
    lastTrialForNotebook = trial + 10
    
    while trial < lastTrialForNotebook:
        convFilters = choose(possibleConvFilters)

        convAndPoolLayers = choose(possibleConvAndPoolLayers)
        convAndPoolLayers = min([convAndPoolLayers,6])

        numberOfFCLayers = choose(possibleNumberOfFCLayers)
        numberOfNeuronsPerFCLayer = choose(possibleNumberOfNeuronsPerFCLayer)

        numberOfEpochs = choose(possibleNumberOfEpochs)
        
        dropoutRate = np.random.rand()
        adamLearningRate = generateAdamLearningRate()
        L2Rate = generateL2()
        

        model = createModel(convFilters, convAndPoolLayers,numberOfFCLayers, numberOfNeuronsPerFCLayer, dropoutRate, adamLearningRate, L2Rate)
    
        model.fit(train,epochs=numberOfEpochs,verbose=0)

        model_path = f'../Models/FlipDetectionModelTrials/fd_model_{trial}.h5'
        model.save(model_path)
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        if model_size < 40:
            print()
            print('trainScore')
            trainScore = model.evaluate(train)[1]
            print('devScore')
            devScore = model.evaluate(dev)[1]

            n_convFilters.append(convFilters)
            n_convAndPoolLayers.append(convAndPoolLayers)
            n_FCLayers.append(numberOfFCLayers)
            n_NeuronsPerFCLayers.append(numberOfNeuronsPerFCLayer)
            n_Epochs.append(numberOfEpochs)
            adamLearningRates.append(adamLearningRate)
            dropoutRates.append(dropoutRate)
            L2Rates.append(L2Rate)
            trainScores.append(trainScore)
            devScores.append(devScore)
            
            print('concluding trial ',trial)
            trial += 1
        else:
            print(f'redoing trial {trial}. Model was {model_size}MB.')
            failedTrial = createModelParametersDF([convFilters],[convAndPoolLayers],
                                                  [numberOfFCLayers],[numberOfNeuronsPerFCLayer],[numberOfEpochs],
                                                  [dropoutRate],[adamLearningRate],[L2Rate],[np.nan],[np.nan])
            display(failedTrial)
            
    modelParametersDF = createModelParametersDF(n_convFilters,n_convAndPoolLayers,
                                                n_FCLayers,n_NeuronsPerFCLayers,n_Epochs,
                                                dropoutRates,adamLearningRates,L2Rates,trainScores,devScores)
    display(modelParametersDF.sort_values(by='trainScore', ascending=False))
        
    
    


# In[11]:

if __name__ == '__main__':
    main()


# In[ ]:




