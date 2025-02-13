# Use an Ubuntu base image
FROM ubuntu:jammy

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install MongoDB
RUN apt-get update && apt-get install -y \
    gnupg curl ca-certificates && \
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/mongodb.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org.list && \
    apt-get update && \
    apt-get install -y mongodb-org && \
    apt-get clean

# Create a directory for MongoDB data
RUN mkdir -p /data/db

# Expose the MongoDB default port
EXPOSE 27017

# Start MongoDB when the container runs
CMD ["mongod", "--bind_ip", "0.0.0.0"]
