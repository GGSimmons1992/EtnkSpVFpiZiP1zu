# MonReader
MonReader is a new mobile document digitization experience for the blind, researchers, and everyone else in need of fully automatic, highly fast, and high-quality document scanning in bulk. It detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops and de-warps it accordingly. 

For this project, the MonReader team collected page-flipping videos from smartphones and labeled them as flipping and not flipping. The goal is to create a CNN from scratch that achieves an F1 score of 0.91 and then use transfer learning and fine-tune the model using Vgg-16, RestNet, and MobileNet.
