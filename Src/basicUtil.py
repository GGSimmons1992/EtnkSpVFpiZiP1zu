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


def createModel(convFilters1, convFilters2, convFilters3, convFilters4, numberOfFCLayers, numberOfNeuronsPerFCLayer, dropoutRate, adamLearningRate,L2Rate):
    model = keras.Sequential()
        
    #convPool1
    model.add(keras.layers.Conv2D(convFilters1,(3,3), activation='relu',padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='valid'))
    #convPool2
    model.add(keras.layers.Conv2D(convFilters2,(3,3), activation='relu',padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='valid'))
    #convPool3
    model.add(keras.layers.Conv2D(convFilters3,(3,3), activation='relu',padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='valid'))
    #finalConv
    model.add(keras.layers.Conv2D(convFilters4,(3,3), activation='relu',padding='valid'))
    
    model.add(keras.layers.Flatten())
    
    for layer in range(numberOfFCLayers):
        if layer == numberOfFCLayers - 1:
            model.add(keras.layers.Dense(1,activation='sigmoid'))
        else:
            model.add(keras.layers.Dense(numberOfNeuronsPerFCLayer,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2Rate)))
            model.add(keras.layers.Dropout(dropoutRate))
            
    adamOptimizer = keras.optimizers.legacy.Adam(learning_rate=adamLearningRate)       
    model.compile(optimizer="adam",loss='binary_crossentropy', metrics=f1_score)
    return model


# In[5]:


def createModelParametersDF(n_convFilters1,n_convFilters2,n_convFilters3,n_convFilters4,
                            n_FCLayers,n_NeuronsPerFCLayers,n_Epochs,
                            dropoutRates,adamLearningRates,L2Rates,trainScores,devScores):
    modelParameters = dict()
    modelParameters['n_convFilters1'] = n_convFilters1
    modelParameters['n_convFilters2'] = n_convFilters2
    modelParameters['n_convFilters3'] = n_convFilters3
    modelParameters['n_convFilters4'] = n_convFilters4
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


def generateDropoutRate(minVal=0,maxVal=1):
    dropoutRate = np.random.random() * (maxVal - minVal) + minVal
    dropoutRate = np.max([dropoutRate,0])
    dropoutRate = np.min([dropoutRate,0.999])
    return dropoutRate


# In[8]:


def generateAdamLearningRate(minVal=1e-4,maxVal=1e-2):
    minVal = np.log10(minVal)
    maxVal = np.log10(maxVal)
    learningRatePower = np.random.random() * (maxVal - minVal) + minVal
    learningRate = np.power(10,learningRatePower)
    return learningRate


# In[9]:


def generateL2(minVal=1e-2,maxVal=1e3):
    minVal = np.log10(minVal)
    maxVal = np.log10(maxVal)
    l2Power = np.random.random() * (maxVal - minVal) + minVal
    l2 = np.power(10,l2Power)
    return l2


# In[10]:


def calculateCriticalPoints(top5ParamList):
    lowPoint = np.min(top5ParamList)
    highPoint = np.max(top5ParamList)
    return (lowPoint,highPoint)


# In[11]:


def calculateLogisticCriticalPoints(top5ParamList):
    top5Log10ParamList = np.log10(top5ParamList)
    log10criticalPointTuple = calculateCriticalPoints(top5Log10ParamList)
    criticalPointTuple = (np.power(10,log10criticalPointTuple[0]),np.power(10,log10criticalPointTuple[1]))
    return criticalPointTuple


# In[12]:


def getAdjustedRange(top5ParamList):
    lowerValue = int(np.max([1,np.min(top5ParamList)]))
    upperValue = int(np.max(top5ParamList))
    
    if lowerValue == upperValue:
        return createRangeFromMidpoint(lowerValue,2*lowerValue)
    return np.arange(lowerValue,upperValue)


# In[13]:


def displayFinalResults(parameterFileName):
    if (exists(parameterFileName)):
        with open(parameterFileName) as d:
            finalResults = json.load(d)
            resultsDictionary = dict()
            for key in finalResults.keys():
                resultsDictionary[key] = [finalResults[key]]
            resultsDF = pd.DataFrame(resultsDictionary,columns = finalResults.keys())
            print('Final Model')
            display(resultsDF)


# In[14]:


def main():

    possibleConvFilters1 = createRangeFromMidpoint(32,64)
    possibleConvFilters2 = createRangeFromMidpoint(32,64)
    possibleConvFilters3 = createRangeFromMidpoint(32,64)
    possibleConvFilters4 = createRangeFromMidpoint(32,64)

    possibleNumberOfFCLayers = createRangeFromMidpoint(10,20)
    possibleNumberOfNeuronsPerFCLayer = createRangeFromMidpoint(10,20)

    possibleNumberOfEpochs = createRangeFromMidpoint(10,20)
    dropoutCriticalPoints = (0,1)
    adamLearningRateCriticalPoints = (1e-4,1e-2)
    L2CriticalPoints = (1e-2,1e3)    
    
    trial = 0
    bestDevScore = 0
    
    train = loadData("training")
    dev,test = loadData("testing",.5)

    choose = np.random.choice

    n_convFilters1 = []
    n_convFilters2 = []
    n_convFilters3 = []
    n_convFilters4 = []
    n_FCLayers = []
    n_NeuronsPerFCLayers = []
    n_Epochs = []
    dropoutRates = []
    adamLearningRates = []
    L2Rates = []
    trainScores = []
    devScores = []
    
    while trial < 100:
        convFilters1 = choose(possibleConvFilters1)
        convFilters2 = choose(possibleConvFilters2)
        convFilters3 = choose(possibleConvFilters3)
        convFilters4 = choose(possibleConvFilters4)

        numberOfFCLayers = choose(possibleNumberOfFCLayers)
        numberOfNeuronsPerFCLayer = choose(possibleNumberOfNeuronsPerFCLayer)

        numberOfEpochs = choose(possibleNumberOfEpochs)
        
        dropoutRate = generateDropoutRate(dropoutCriticalPoints[0],dropoutCriticalPoints[1])
        adamLearningRate = generateAdamLearningRate(adamLearningRateCriticalPoints[0],adamLearningRateCriticalPoints[1])
        L2Rate = generateL2(L2CriticalPoints[0],L2CriticalPoints[1])
        

        model = createModel(convFilters1, convFilters2, convFilters3, convFilters4,
                            numberOfFCLayers, numberOfNeuronsPerFCLayer, dropoutRate, adamLearningRate,L2Rate)
    
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

            if (devScore > 0.91) and (devScore > bestDevScore):
                testScore = model.evaluate(test)[1]
                model_path = f'../Models/best_fd_model_.h5'
                model.save(model_path)
                bestModelParams = {
                    'n_convFilters1' : int(convFilters1),
                    'n_convFilters2' : int(convFilters2),
                    'n_convFilters3' : int(convFilters3),
                    'n_convFilters4' : int(convFilters4),
                    'n_FCLayers' : int(numberOfFCLayers),
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
                with open('../Models/best_fd_model_params.json', 'w') as f:
                    json.dump(bestModelParams, f)
                bestDevScore = devScore       

            n_convFilters1.append(convFilters1)
            n_convFilters2.append(convFilters2)
            n_convFilters3.append(convFilters3)
            n_convFilters4.append(convFilters4)
            
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
            failedTrial = createModelParametersDF([convFilters1],[convFilters2],[convFilters3],[convFilters4],
                                                  [numberOfFCLayers],[numberOfNeuronsPerFCLayer],[numberOfEpochs],
                                                  [dropoutRate],[adamLearningRate],[L2Rate],[np.nan],[np.nan])
            display(failedTrial)
        
            
        if (trial % 10 == 9): 
            modelParametersDF = createModelParametersDF(n_convFilters1,n_convFilters2,n_convFilters3,n_convFilters4,
                                                    n_FCLayers,n_NeuronsPerFCLayers,n_Epochs,
                                                    dropoutRates,adamLearningRates,L2Rates,trainScores,devScores)
            modelParametersDF = modelParametersDF.sort_values(by='trainScore', ascending=False)
            display(modelParametersDF)
            
            top5 = modelParametersDF[0:5]
            possibleConvFilters1 = getAdjustedRange(top5['n_convFilters1'])
            possibleConvFilters2 = getAdjustedRange(top5['n_convFilters2'])
            possibleConvFilters3 = getAdjustedRange(top5['n_convFilters3'])
            possibleConvFilters4 = getAdjustedRange(top5['n_convFilters4'])
        
            possibleNumberOfFCLayers = getAdjustedRange(top5['n_FCLayers'])
            possibleNumberOfNeuronsPerFCLayer = getAdjustedRange(top5['n_NeuronsPerFCLayers'])
        
            possibleNumberOfEpochs = getAdjustedRange(top5['n_Epochs'])
            dropoutCriticalPoints = calculateCriticalPoints(top5['dropoutRate'])
            adamLearningRateCriticalPoints = calculateLogisticCriticalPoints(top5['adamLearningRates'])
            L2CriticalPoints = calculateLogisticCriticalPoints(top5['L2Rates'])

            n_convFilters1 = []
            n_convFilters2 = []
            n_convFilters3 = []
            n_convFilters4 = []
            n_FCLayers = []
            n_NeuronsPerFCLayers = []
            n_Epochs = []
            dropoutRates = []
            adamLearningRates = []
            L2Rates = []
            trainScores = []
            devScores = []

            if bestDevScore > 0.91:
                trial = 101
                
    displayFinalResults('../Models/best_fd_model_params.json')
            
    


# In[15]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




