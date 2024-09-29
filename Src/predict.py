import numpy as np
import transferUtil
import basicUtil
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import custom_object_scope
import os
import io
import IPython.display
from PIL import Image
import gradio as gr
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

imageShape = (180,180)
inputShape = [imageShape[0],imageShape[1],3]

with custom_object_scope({'f1_score':basicUtil.f1_score}):
    model = keras.models.load_model('../Models/best_mobilenet_model_.h5')
model.summary()

def preprocess_image(image, imageShape=(180,180)):
    if isinstance(image, Image.Image):
        image = image.resize(imageShape)
    else:
        image = Image.fromarray(image).resize(imageShape)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def flipDetector(image):
    processedImage = preprocess_image(image)
    result = model.predict(processedImage)
    print(result[0,0])
    return "flipped" if (result[0,0] > .5) else "not flipped"

def main():

    gr.close_all()
    demo = gr.Interface(fn=flipDetector,
                        inputs=[gr.Image(label="Upload image", type="pil")],
                        outputs=[gr.Textbox(label="Flip status")],
                        title="MonReader Flip Detector",
                        description="Detect page flipping status using MonReader Effecient Net model",
                        allow_flagging="never",
                        examples=["../Data/training/notflip/0001_000000001.jpg", "../Data/training/flip/0001_000000010.jpg"])

    demo.launch(share=True, server_port=int(os.environ['PORT1']))

if __name__ == "__main__":
    main()