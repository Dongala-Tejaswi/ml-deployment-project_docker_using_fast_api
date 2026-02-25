# ml-deployment-project_docker_using_fast_api
Training phase:
Dataset → Train model → Save → model.joblib

Deployment phase:
model.joblib → Load → FastAPI → Prediction

User sends input
      ↓
FastAPI receives input
      ↓
predict() function runs
      ↓
model predicts result
      ↓
FastAPI returns output
