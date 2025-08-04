# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8050

# Start Gunicorn server with environment variables
CMD ["gunicorn", "dashboard:server", "--workers", "4", "--bind", "0.0.0.0:8050"]
