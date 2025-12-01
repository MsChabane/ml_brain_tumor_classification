from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve, auc
import numpy as np
from json import dump
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model,X_test,y_test):
    proba=model.predict_proba(X_test)[:,1]
    ypreds=model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, proba)
    result={
    'accuracy':accuracy_score(y_test,ypreds),
    'precision':precision_score(y_test,ypreds),
    'recall':recall_score(y_test,ypreds),
    'f1_score':f1_score(y_test,ypreds),
    'auc':auc(fpr, tpr)
    }
    with open('./reports/ml/result.json','w') as f:
        dump(result,f)
    
    cm = confusion_matrix(y_test, ypreds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('./reports/ml/confusion_matrix.png')
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('./reports/ml/roc_curve.png')



