FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install them
#COPY survey-app/requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY survey-app/. .

EXPOSE 5001

CMD ["python", "app.py"]
