import cv2
import os
import matplotlib.pyplot as plt
from LBPNET import *
from train import train

def load_images_from_folder(folder):
    images = []
    labels = []
    iterator = 0
    for subfolder in os.listdir(folder):
        for filename in os.listdir(os.path.join(folder,subfolder)):
         img = os.path.join(folder,subfolder,filename)
         images.append(img)
         labels.append(iterator)
        iterator = iterator + 1
    return images, labels

cale = r'lfw2/lfw2'
images_path, labels = load_images_from_folder(cale)
images=[]
for path in images_path:
    img = plt.imread(path)
    images.append(img)

# labels = []
# for i in range(5749):
#    labels.append(i)
# Create an instance of the LBPNet class
model = LBPNet(num_classes=5749)

model_trained = train(model, 100, images, labels)


