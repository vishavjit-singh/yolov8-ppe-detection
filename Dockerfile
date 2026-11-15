# Serving image: ONNX Runtime + OpenCV + the compliance engine. No PyTorch, no Ultralytics.
FROM python:3.11-slim-bookworm

ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 PPE_MODEL=/models/best.onnx
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
RUN useradd --create-home app
USER app
EXPOSE 8000
# Mount a model trained by the train workflow:  docker run -v $PWD/runs/ppe/weights:/models -p 8000:8000 ppe
CMD ["uvicorn", "ppe.api:app", "--host", "0.0.0.0", "--port", "8000"]
