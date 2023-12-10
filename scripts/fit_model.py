from catboost import Pool, CatBoostClassifier
import numpy as np
import pickle
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.set_experiment("model_preparation")
with mlflow.start_run(experiment_id=experiment.experiment_id):

    with open('/home/vmac/lab3/dataset/X_train.npy', 'rb') as f:    
        X_train = np.load(f, allow_pickle=True)
    with open('/home/vmac/lab3/dataset/X_test.npy', 'rb') as f:    
        X_test= np.load(f, allow_pickle=True)
    with open('/home/vmac/lab3/dataset/y_train.npy', 'rb') as f:    
        y_train= np.load(f, allow_pickle=True)
    with open('/home/vmac/lab3/dataset/y_test.npy', 'rb') as f:    
        y_test= np.load(f, allow_pickle=True)

    learning_rate = 0.1
    depth = 4
    l2_leaf_reg = 1
    mlflow.log_param('learning_rate',learning_rate)
    mlflow.log_param('depth',depth)
    mlflow.log_param('l2_leaf_reg',l2_leaf_reg)
   
    train_dataset = Pool(X_train, y_train)
    test_dataset = Pool(X_test)

    model = CatBoostClassifier(
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg
        )
    model.fit(train_dataset)
    mlflow.sklearn.log_model(model, "")
    with open("/home/vmac/lab3/model/model.pkl", "wb") as m:    
        pickle.dump(model, m)