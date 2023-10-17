from PIL import Image
import numpy as np
import os
from simple_nn import SimpleNN
import matplotlib.pyplot as plt

def print_plot(accuracy_history, loss_history):
    plt.subplot(2, 1, 1)
    plt.plot(accuracy_history)
    plt.title('Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss_history)
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            img = img.convert('L')
            img = np.array(img).flatten()
            img = img / 255.0
            images.append(img)
            label = int(filename.split("-num")[-1].split(".")[0])
            labels.append(label)
    return np.array(images), np.array(labels)


nn = SimpleNN(784, 50, 10)
images, labels = load_images("img")
accuracy_history, loss_history = nn.train(images, labels, num_passes=10000, epsilon=0.01, batch_size=100)
if not os.path.exists("../weights"):
    os.makedirs("../weights")
np.save('../weights/W1.npy', nn.W1)
np.save('../weights/b1.npy', nn.b1)
np.save('../weights/W2.npy', nn.W2)
np.save('../weights/b2.npy', nn.b2)
print("Model trained and weights saved.")
print_plot(accuracy_history, loss_history)
