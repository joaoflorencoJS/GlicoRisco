# Compatível com Hugging Face Spaces (SDK Docker, porta 7860) e Render (usa $PORT)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py predictor.py ./
COPY templates ./templates
COPY models ./models
ENV PORT=7860
EXPOSE 7860
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 60
