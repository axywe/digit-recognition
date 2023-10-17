import os
from PIL import Image
import numpy as np
import json
import sys
from simple_nn import SimpleNN
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


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


def evaluate_model(model, X_test, y_test):
    predictions = np.argmax(model.forward(X_test), axis=1)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':
    X, y = load_images("img")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    for hidden_layer_size in [10, 50, 100]:
        sys.stdout.write(f"Testing model with hidden layer size: {hidden_layer_size}\n")

        model = SimpleNN(784, hidden_layer_size, 10)

        sys.stdout.write("Training model...\n")
        model.train(X_train, y_train, num_passes=1000, epsilon=0.01, batch_size=32)

        sys.stdout.write("Evaluating model...\n")
        metrics = evaluate_model(model, X_test, y_test)

        results[str(hidden_layer_size)] = metrics
        sys.stdout.write(f"Results: {metrics}\n")

    with open('new_objective_test_results.json', 'w') as f:
        json.dump(results, f)
    sys.stdout.write("Saved results to new_objective_test_results.json\n")
