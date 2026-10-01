FROM apache/airflow:2.8.4-python3.11

USER airflow

COPY --chown=airflow:root requirements-airflow.txt /tmp/requirements-airflow.txt

RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.4/constraints-3.11.txt"

RUN mkdir -p /opt/airflow/project/models /opt/airflow/project/artifacts

ENV PYTHONPATH=/opt/airflow/project
