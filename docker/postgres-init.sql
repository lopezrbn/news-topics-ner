-- docker/postgres-init.sql
-- Se ejecuta como usuario ${POSTGRES_USER} (postgres) la primera vez que arranca el contenedor.

-----------------------------
-- news_nlp (negocio)
-----------------------------
CREATE ROLE news_nlp WITH LOGIN PASSWORD 'news_nlp_password';
CREATE DATABASE news_nlp OWNER news_nlp;
GRANT ALL PRIVILEGES ON DATABASE news_nlp TO news_nlp;

\connect news_nlp;
GRANT ALL ON SCHEMA public TO news_nlp;

\connect postgres;

-----------------------------
-- Airflow
-----------------------------
CREATE ROLE airflow_user WITH LOGIN PASSWORD 'airflow_password';
CREATE DATABASE airflow_db OWNER airflow_user;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

\connect airflow_db;
GRANT ALL ON SCHEMA public TO airflow_user;

\connect postgres;

-----------------------------
-- MLflow
-----------------------------
CREATE ROLE mlflow_user WITH LOGIN PASSWORD 'mlflow_password';
CREATE DATABASE mlflow_db OWNER mlflow_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

\connect mlflow_db;
GRANT ALL ON SCHEMA public TO mlflow_user;

\connect postgres;
