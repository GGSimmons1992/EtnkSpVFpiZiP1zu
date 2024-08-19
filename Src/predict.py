import numpy as np
import transferUtil
import basicUtil
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import custom_object_scope

imageShape = (180,180)
inputShape = [imageShape[0],imageShape[1],3]

dataset = transferUtil.loadDataWithResizeShape("testing",imageShape)

#Using Effecient Net
with custom_object_scope({'f1_score':basicUtil.f1_score}):
    model = keras.models.load_model('../Models/best_effecientnet_model_.h5')
results = model.evaluate(dataset)
for result in results:
    print(result)

predictions = model.predict(dataset)
predictions = tf.where(predictions < 0.5,0,1)
print(predictions.numpy())
