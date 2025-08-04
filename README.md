# MonReader
MonReader is a new mobile document digitization experience for the blind, researchers, and everyone else needing fully automatic, highly fast, and high-quality document scanning in bulk. It detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops and de-warps it accordingly. 

For this project, the MonReader team collected page-flipping videos from smartphones and labeled them as flipping and not flipping. The goal is to create a CNN from scratch that achieves an F1 score of 0.91 and then use transfer learning and fine-tune the model using Vgg-16, ResNet, and MobileNet. The model must be under 40MB

Our final model is a MobileNet model with a size of 6.02MB and a test f1 score of 0.99. This model is utilized in predict.py which is utilized by a grad.io app.

## Contents

### Data
The data is provided by Apziva consisting of a training folder and a testing folder. Inside both folders, there are jpg files of notflip and yesflip images.

### Models
The Models folder contains .gitignored folders of EffecientNetTrials, FlipDetectionModelTrials, MobileNetTrials, and VGG16Trials. These trial folders contain the models of all attempted models. The best model is re-copied to the Models folder, where it is visible and not hidden by .gitignore. The parameters of those models are also visible.

### Notebooks
Each notebook shows how each model listed above is trained. FlipDetectionModelV1.ipynb is a raw Keras Sequential model using convolution and pooling layers. The functions in FlipDetectionModelV1.ipynb are nb converted into basicUtils.py in the Src folder. The other four models are transfer learning models. Function in VGG16.ipynb are nb converted into transferUtils.py in the Src folder. The other three transfer learning notebooks reference the functions in transferUtils.py.

### Src
The Src folder holds basicUtils.py, which has functions needed by all models to create and train them. It also holds transferUtils.py that has functions specifically used for the transfer learning models. Lastly, it contains predict.py which one can run a grad.io app using the mobile net model to upload a flipped or not flipped image of a book to identify if the book page is flipped or not flipped.

## Requirements.txt
list of Python packages used by this repo.

## LICENSE
This project uses an MIT license
