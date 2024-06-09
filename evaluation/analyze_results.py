from typing import List, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, confusion_matrix, classification_report


def calculate_metrics(true_labels: List[int], predictions: List[int], class_names: List[str], probabilities_list: List[np.ndarray]) -> Tuple[float, float, float, np.ndarray]:
    """
    Calculate precision, recall, F1-score, and confusion matrix.

    Parameters:
    - true_labels (List[int]): True labels.
    - predictions (List[int]): Predicted labels.
    - class_names (List[str]): Class names.
    - probabilities_list (List[np.ndarray]): List of prediction probabilities.

    Returns:
    - precision (float): Weighted precision.
    - recall (float): Weighted recall.
    - f1 (float): Weighted F1-score.
    - conf_matrix (np.ndarray): Confusion matrix.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)
    print(classification_report(true_labels, predictions, target_names=class_names))
    return precision, recall, f1, conf_matrix


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str]) -> None:
    """
    Plot Confusion Matrix.

    Parameters:
    - conf_matrix (np.ndarray): Confusion matrix.
    - class_names (List[str]): Class names.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_precision_recall_curve(true_labels: List[int], probabilities_list: List[np.ndarray], class_names: List[str]) -> None:
    """
    Plot Precision-Recall Curve.

    Parameters:
    - true_labels (List[int]): True labels.
    - probabilities_list (List[np.ndarray]): List of prediction probabilities.
    - class_names (List[str]): Class names.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        y_true = [1 if label == i else 0 for label in true_labels]
        y_scores = [prob[i] for prob in probabilities_list]
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, marker='.', label=class_name)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


def save_classification_report(true_labels: List[int], predictions: List[int], class_names: List[str]) -> None:
    """
    Create and Save Classification Report.

    Parameters:
    - true_labels (List[int]): True labels.
    - predictions (List[int]): Predicted labels.
    - class_names (List[str]): Class names.

    Returns:
    - None
    """
    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    report_df.to_csv('classification_report.csv')


def analyze_results(true_labels: List[int], predictions: List[int], class_names: List[str], probabilities_list: List[np.ndarray]) -> None:
    """
    Perform result analysis.

    Parameters:
    - true_labels (List[int]): True labels.
    - predictions (List[int]): Predicted labels.
    - class_names (List[str]): Class names.
    - probabilities_list (List[np.ndarray]): List of prediction probabilities.

    Returns:
    - None
    """
    precision, recall, f1, conf_matrix = calculate_metrics(true_labels, predictions, class_names, probabilities_list)
    plot_confusion_matrix(conf_matrix, class_names)
    plot_precision_recall_curve(true_labels, probabilities_list, class_names)
    save_classification_report(true_labels, predictions, class_names)
