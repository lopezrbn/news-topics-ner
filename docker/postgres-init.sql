-- Crear usuario y base de datos de negocio (news_nlp)
CREATE USER news_nlp_user WITH PASSWORD 'news_nlp_password';
CREATE DATABASE news_nlp OWNER news_nlp_user;
GRANT ALL PRIVILEGES ON DATABASE news_nlp TO news_nlp_user;

-- Crear usuario y base de datos de Airflow
CREATE USER airflow_user WITH PASSWORD 'airflow_password';
CREATE DATABASE airflow_db OWNER airflow_user;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

-- Crear usuario y base de datos de MLflow
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
CREATE DATABASE mlflow_db OWNER mlflow_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
