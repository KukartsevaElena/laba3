from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    "owner": "Admin",
    "start_date": dt.datetime(2023, 11, 26),
    "retries": 5,
    "retry_delays": dt.timedelta(minutes=5),
    "depends_on_past": False
}

with DAG(
    dag_id='pipeline',    
    default_args=args,
    schedule_interval=None,    
    tags=['pipeline', 'score'],
) as dag:    
    data_create = BashOperator(task_id='data_create',
    bash_command="python3 /home/vmac/lab3/scripts/data_create.py",    
    dag=dag)
    data_preprocessing = BashOperator(task_id='data_preprocessing',    
                                       bash_command="python3 /home/vmac/lab3/scripts/data_preprocessing.py",
    dag=dag)    
    fit_model = BashOperator(task_id='fit_model',
    bash_command="python3 /home/vmac/lab3/scripts/fit_model.py",    
    dag=dag)
    predict = BashOperator(task_id='predict',    
                                 bash_command="python3 /home/vmac/lab3/scripts/predict.py",
    dag=dag)   
    data_create >> data_preprocessing >> fit_model >> predict