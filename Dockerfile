FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY analysis.py .
COPY exploratory_data_analysis.py .
COPY dating_app_behavior_dataset.csv .

CMD ["python", "analysis.py"]