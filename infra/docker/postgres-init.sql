-- Run automatically on first postgres container start.
-- Creates the mlflow tracking database alongside the main app DB.

CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO authentiguard;
