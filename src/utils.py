from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


import matplotlib.pyplot as plt


LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def save_confusion_matrix(true_labels, predicted_labels, filename):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax, cmap='Blues')
    plt.xticks(rotation=45)
    plt.savefig(filename)
