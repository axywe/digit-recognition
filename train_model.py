from PIL import Image
import numpy as np
import os
from simple_nn import SimpleNN
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def print_plot(accuracy_history, loss_history):
    pp = PdfPages('test.pdf')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(accuracy_history, linewidth=2, c="blue")
    plt.plot(loss_history, linewidth=2, c="red")
    plt.title('Accuracy and Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=-1)
    plt.xlim(right=len(accuracy_history) + 1)
    plt.show()
    pp.savefig(fig)
    pp.close()


def load_images(folder):
    print("Loading data...")
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


nn = SimpleNN(784, 100, 10)
images, labels = load_images("img")
accuracy_history, loss_history = nn.train(images, labels, num_passes=10000, epsilon=0.01, batch_size=100)
if not os.path.exists("weights"):
    os.makedirs("weights")
np.save('weights/W1.npy', nn.W1)
np.save('weights/b1.npy', nn.b1)
np.save('weights/W2.npy', nn.W2)
np.save('weights/b2.npy', nn.b2)
print("Model trained and weights saved.")
print_plot(accuracy_history, loss_history)
