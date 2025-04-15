FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Copy code
COPY python_src python_src
WORKDIR python_src

# Install python
RUN apt update && apt install python3 python3-pip -y

# Install deps
RUN pip3 install -r ./requirements.txt

# Update to read pyproject
RUN pip3 install --upgrade pip

# Install transferwareai
RUN pip3 install .

WORKDIR scripts
EXPOSE 8080:8080
# Run query api. Note that the settings file is compiled into container
CMD ["python3", "query_api.py"]