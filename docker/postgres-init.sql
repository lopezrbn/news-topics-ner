    -- news_nlp (proyecto)
CREATE USER news_nlp_user WITH PASSWORD 'news_nlp_password';
CREATE DATABASE news_nlp OWNER news_nlp_user;

-- mlflow
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
CREATE DATABASE mlflow_db OWNER mlflow_user;

-- airflow
CREATE USER airflow_user WITH PASSWORD 'airflow_password';
CREATE DATABASE airflow_db OWNER airflow_user;