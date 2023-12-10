import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import pickle
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.set_experiment("predict")
with mlflow.start_run(experiment_id=experiment.experiment_id):

    model = pickle.load(open('/home/vmac/lab3/model/model.pkl','rb'))
    with open('/home/vmac/lab3/dataset/y_test.npy', 'rb') as f:    
        y_test= np.load(f, allow_pickle=True)
    with open('/home/vmac/lab3/dataset/X_test.npy', 'rb') as f:    
        X_test= np.load(f, allow_pickle=True)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred), f1_score(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, average="micro")    
    mlflow.log_metric("F1_score", float(f1))
