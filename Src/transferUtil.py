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
import json
from os.path import exists
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from numpy.random import choice as choose

import sys
sys.path.insert(0, "../Src/")
import basicUtil




# In[2]:


def loadDataWithResizeShape(dataset,resizeShape,split=0):
    if(split == 0):
        print("In non split")
        return tf.keras.preprocessing.image_dataset_from_directory(f"../Data/{dataset}",
                                                                   labels='inferred',shuffle=True,seed=51,
                                                                   image_size=resizeShape, batch_size=32)
    else:
        print("In split")
        return tf.keras.preprocessing.image_dataset_from_directory(f"../Data/{dataset}",
                                                                   labels='inferred',shuffle=True,seed=51,
                                                                   validation_split=split,subset='both',
                                                                   image_size=resizeShape, batch_size=32)


# In[3]:


def createTransferableModel(base_model, numberOfNeuronsPerFCLayer, dropoutRate, adamLearningRate, L2Rate):
    model = keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(numberOfNeuronsPerFCLayer,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2Rate)))
    model.add(keras.layers.Dropout(dropoutRate))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
            
    adamOptimizer = keras.optimizers.legacy.Adam(learning_rate=adamLearningRate)       
    model.compile(optimizer=adamOptimizer,loss='binary_crossentropy', metrics=basicUtil.f1_score)
    return model


# In[4]:


def createTransferModelParametersDF(n_NeuronsPerFCLayers,n_Epochs,
                            dropoutRates,adamLearningRates,L2Rates,trainScores,devScores):
    modelParameters = dict()
    modelParameters['n_NeuronsPerFCLayers'] = n_NeuronsPerFCLayers
    modelParameters['n_Epochs'] = n_Epochs
    modelParameters['dropoutRate'] = dropoutRates
    modelParameters['adamLearningRates'] = adamLearningRates
    modelParameters['L2Rates'] = L2Rates
    modelParameters['trainScore'] = trainScores
    modelParameters['devScore'] = devScores

    modelParametersDF = pd.DataFrame(modelParameters, columns=modelParameters.keys())
    return modelParametersDF


# In[5]:


def main():
    imageShape = (180,180)
    inputShape = [imageShape[0],imageShape[1],3]
    train = loadDataWithResizeShape("training",imageShape)
    dev,test = loadDataWithResizeShape("testing",imageShape,.5)
    
    base_model = VGG16(weights="imagenet", include_top=False,input_shape=inputShape,
                      classifier_activation = None,classes = len(train.class_names))
    for layer in base_model.layers:
        layer.trainable = False

    possibleNeuronsPerLayer = basicUtil.createRangeFromMidpoint(16,32)
    possibleEpochs = basicUtil.createRangeFromMidpoint(5,10)

    dropoutCriticalPoints = (0,1)
    adamLearningRateCriticalPoints = (1e-4,1e-2)
    L2CriticalPoints = (1e-2,1e3) 

    n_NeuronsPerFCLayers = []
    n_Epochs = []
    dropoutRates = []
    adamLearningRates = []
    L2Rates = []
    trainScores = []
    devScores = []

    trial = 0
    bestDevScore = 0
    oversizedNeuronNumbers = []
    oversizedEpochNumbers = []
    
    while trial < 100:
        numberOfNeuronsPerFCLayer = choose(possibleNeuronsPerLayer)
        numberOfEpochs = choose(possibleEpochs)
        
        dropoutRate = basicUtil.generateDropoutRate(dropoutCriticalPoints[0],dropoutCriticalPoints[1])
        adamLearningRate = basicUtil.generateAdamLearningRate(adamLearningRateCriticalPoints[0],adamLearningRateCriticalPoints[1])
        L2Rate = basicUtil.generateL2(L2CriticalPoints[0],L2CriticalPoints[1])
        model = createTransferableModel(base_model, numberOfNeuronsPerFCLayer, dropoutRate, adamLearningRate, L2Rate)
        
        model.fit(train,epochs=numberOfEpochs,verbose=0)

        model_path = f'../Models/VGG16Trials/vgg16_model_{trial}.h5'
        model.save(model_path)
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        if model_size < 60:
            oversizedNeuronNumbers = []
            oversizedEpochNumbers = []
            print('trainScore')
            trainScore = model.evaluate(train)[1]
            print('devScore')
            devScore = model.evaluate(dev)[1]

            if (devScore > 0.91) and (devScore > bestDevScore):
                testScore = model.evaluate(test)[1]
                model_path = f'../Models/best_vgg16_model_.h5'
                model.save(model_path)
                bestModelParams = {
                    'n_NeuronsPerFCLayers' : int(numberOfNeuronsPerFCLayer),
                    'n_Epochs' : int(numberOfEpochs),
                    'dropoutRate' : dropoutRate,
                    'adamLearningRates' : adamLearningRate,
                    'L2Rates' : L2Rate,
                    'modelSize' : model_size,
                    'trainScore': trainScore,
                    'devScore': devScore,
                    'testScore': testScore
                }
                with open('../Models/best_vgg16_model_params.json', 'w') as f:
                    json.dump(bestModelParams, f)
                bestDevScore = devScore       

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
            failedTrial = createTransferModelParametersDF([numberOfNeuronsPerFCLayer],[numberOfEpochs],
                                                  [dropoutRate],[adamLearningRate],[L2Rate],[np.nan],[np.nan])
            display(failedTrial)
            oversizedNeuronNumbers.append(numberOfNeuronsPerFCLayer)
            oversizedEpochNumbers.append(numberOfEpochs)
            if len(oversizedNeuronNumbers) >= 3:
                possibleNeuronsPerLayer = [x for x in possibleNeuronsPerLayer if x < np.mean(oversizedNeuronNumbers)]
                possibleEpochs = [x for x in possibleEpochs if x < np.mean(oversizedEpochNumbers)]
                oversizedNeuronNumbers = []
                oversizedEpochNumbers = []
            
        if (trial % 10 == 9): 
            modelParametersDF = createTransferModelParametersDF(n_NeuronsPerFCLayers,n_Epochs,
                                                    dropoutRates,adamLearningRates,L2Rates,trainScores,devScores)
            modelParametersDF = modelParametersDF.sort_values(by='trainScore', ascending=False)
            display(modelParametersDF)
            
            top5 = modelParametersDF[0:5]
            possibleNumberOfNeuronsPerFCLayer = basicUtil.getAdjustedRange(top5['n_NeuronsPerFCLayers'])
        
            possibleNumberOfEpochs = basicUtil.getAdjustedRange(top5['n_Epochs'])
            dropoutCriticalPoints = basicUtil.calculateCriticalPoints(top5['dropoutRate'])
            adamLearningRateCriticalPoints = basicUtil.calculateLogisticCriticalPoints(top5['adamLearningRates'])
            L2CriticalPoints = basicUtil.calculateLogisticCriticalPoints(top5['L2Rates'])

            n_NeuronsPerFCLayers = []
            n_Epochs = []
            dropoutRates = []
            adamLearningRates = []
            L2Rates = []
            trainScores = []
            devScores = []

            if bestDevScore > 0.91:
                trial = 101
                
    basicUtil.displayFinalResults('../Models/best_vgg16_model_params.json')
    


# In[6]:


if __name__ == '__main__':
    main()


# In[ ]:




