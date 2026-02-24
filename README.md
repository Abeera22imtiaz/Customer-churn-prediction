Telco Customer Churn Prediction System

-Purpose
Build and ship a comprehensive Machine Learning solution for predicting customer churn in a telecom environment—transitioning from raw data processing and experimental modeling to a production-ready API and web interface.

------Problem Solved & Benefits
-Data-Driven Decisions: Predicts high-risk customers, allowing retention teams to act before a customer leaves.
-Operationalized ML: Moves the model out of static notebooks into a live REST API and a user-friendly UI.
-Reproducible Experiments: Uses MLflow to track every training run, ensuring metrics and artifacts are audit-ready.
-Consistent Delivery: Uses Docker and CI/CD to ensure the application runs identically on any machine or server.

------What I Built
-Modeling & Tracking: Developed an XGBoost classifier with custom feature engineering, logging all runs and serialized models to MLflow.
-Inference Service: A FastAPI application providing a /predict (POST) endpoint and a root health check /.
-Interactive Web UI: A Gradio interface integrated into the FastAPI app, allowing manual testing of the model without writing code.
-Containerization: A multi-layered Dockerfile that packages the code, dependencies, and the "Champion" model into a portable image.
-CI/CD Pipeline: A GitHub Actions workflow that automates the building and pushing of the Docker image to Docker Hub on every push to the main branch.

------Project Structure

.
├── .github/workflows/    # CI/CD pipelines
├── artifacts/            # Global feature schemas
├── src/
│   ├── app/              # FastAPI & Gradio UI logic
│   ├── data/             # Preprocessing & cleaning scripts
│   ├── model/            # Training and evaluation logic
│   └── serving/          # Inference logic & bundled model artifacts
├── Dockerfile            # Container configuration
└── requirements.txt      # Project dependencies

--------Roadblocks & Solutions
-Path Management: Resolved complex directory issues between local Windows paths and Linux Docker paths using os.path.abspath.
-Deployment Errors: Fixed ModuleNotFoundError in Docker by explicitly setting the PYTHONPATH environment variable in the Dockerfile.
-Schema Consistency: Created a dedicated feature alignment step to ensure the API never crashes if the user provides extra or missing data fields.

-----Deployment Flow (High-Level)
-Code Push: Developer pushes changes to the main branch.
-Automated Build: GitHub Actions triggers a build using the Dockerfile.
-Registry Update: The new image is tagged and pushed to Docker Hub (abeera22imtiaz/churn:latest).
-Deployment: The container can be pulled and run on any server with a single command:
docker run -p 8000:8000 abeera22imtiaz/churn:latest