import numpy as np
def get_confusion_matrix(output,target):
    confusion_matrix = np.zeros((output[0].shape[0],output[0].shape[0]))
    for i in range(len(output)):
        true_idx = target[i]
        pred_idx = np.argmax(output[i])
        confusion_matrix[true_idx][pred_idx] += 1.0
    return confusion_matrix

def get_confusion_matrix_logits(output,target):
    confusion_matrix = np.zeros((2,2))
    for i in range(len(output)):
        true_idx = target[i]
        pred_idx = 1 if output[i]>0.5 else 0
        confusion_matrix[true_idx][pred_idx] += 1.0
    return confusion_matrix
