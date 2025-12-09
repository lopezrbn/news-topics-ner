    -- news_nlp (proyecto)
CREATE DATABASE news_nlp;
CREATE USER news_nlp WITH PASSWORD 'news_nlp_password';
GRANT ALL PRIVILEGES ON DATABASE news_nlp TO news_nlp;

-- airflow
CREATE DATABASE airflow_db;
CREATE USER airflow WITH PASSWORD 'airflow_password';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow;

-- mlflow
CREATE DATABASE mlflow_db;
CREATE USER mlflow WITH PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;
