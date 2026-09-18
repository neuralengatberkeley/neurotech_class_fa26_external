import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Linear discriminant classification helpers.

directions = np.arange(1, 9)

def run_leave_one_out_classification(features,labels):
    features = np.asarray(features).reshape(len(labels),-1)
    return cross_val_predict(LinearDiscriminantAnalysis(),features,labels,cv=LeaveOneOut())

def plot_confusion(y_true,y_pred,title,ax=None):
    matrix = confusion_matrix(y_true,y_pred,labels=directions,normalize="true")
    if ax is None:
        _,ax = plt.subplots(figsize=(5,4),dpi=120)
    ConfusionMatrixDisplay(matrix,display_labels=directions).plot(
        ax=ax,cmap="hot",values_format=".2f",colorbar=False,im_kw={"vmin":0,"vmax":1})
    ax.set(xlabel="Predicted target",ylabel="True target",
           title=f"{title}\nAccuracy = {np.mean(y_true==y_pred):.3f}")
    return matrix