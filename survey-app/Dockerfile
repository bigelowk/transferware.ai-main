FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install them
#COPY survey-app/requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY survey-app/. .

#Download Deps
#RUN npm install

#Build App
#RUN npm run build

EXPOSE 5001

CMD ["python", "app.py"]
