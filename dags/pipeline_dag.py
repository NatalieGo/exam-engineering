from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import subprocess
import os

# Укажем пути к скриптам и логам
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../etl'
LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../logs'

os.makedirs(LOG_DIR, exist_ok=True)

default_args = {
    'owner': 'student',
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': 300,  # 5 минут
}

dag = DAG(
    'breast_cancer_pipeline',
    default_args=default_args,
    description='ETL pipeline for Breast Cancer dataset',
    schedule_interval=None,  # отключено расписание
)

def run_script(script_name):
    """
    Запускает указанный скрипт и сохраняет лог
    """
    script_path = os.path.join(BASE_DIR, script_name)
    log_path = os.path.join(LOG_DIR, f"{script_name}.log")

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            ['python3', script_path],
            stdout=log_file,
            stderr=log_file
        )
        process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Script {script_name} failed. See log {log_path}")

# Операторы
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=lambda: run_script('load_data.py'),
    dag=dag,
)

clean_data_task = PythonOperator(
    task_id='clean_data',
    python_callable=lambda: run_script('clean_data.py'),
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=lambda: run_script('train_model.py'),
    dag=dag,
)

calc_metrics_task = PythonOperator(
    task_id='calc_metrics',
    python_callable=lambda: run_script('calc_metrics.py'),
    dag=dag,
)

# Порядок выполнения
load_data_task >> clean_data_task >> train_model_task >> calc_metrics_task
