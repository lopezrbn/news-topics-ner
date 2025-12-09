-- 1) Crear usuarios
CREATE USER news_nlp_user WITH PASSWORD 'news_nlp_password';
CREATE USER mlflow_user   WITH PASSWORD 'mlflow_password';
CREATE USER airflow_user  WITH PASSWORD 'airflow_password';

-- 2) Crear bases de datos
CREATE DATABASE news_nlp   OWNER news_nlp_user;
CREATE DATABASE mlflow_db  OWNER mlflow_user;
CREATE DATABASE airflow_db OWNER airflow_user;

-- 3) Dar permisos completos y cambiar el owner del schema public
--    MLflow
\connect mlflow_db
GRANT ALL ON SCHEMA public TO mlflow_user;
ALTER SCHEMA public OWNER TO mlflow_user;

--    Airflow
\connect airflow_db
GRANT ALL ON SCHEMA public TO airflow_user;
ALTER SCHEMA public OWNER TO airflow_user;

--    Negocio
\connect news_nlp
GRANT ALL ON SCHEMA public TO news_nlp_user;
ALTER SCHEMA public OWNER TO news_nlp_user;
