# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install Git LFS and Git
RUN apt-get update && \
    apt-get install -y git-lfs git && \
    git lfs install

# Set the working directory to /app
WORKDIR /app

# Download the repo and models directory
RUN git clone https://github.com/sammimagg/flask-mlZ6.git && \
    cd flask-mlZ6 && \
    git lfs pull && \
    cd .. && \
    git clone https://github.com/sammimagg/XL-Net-model.git && \
    cd XL-Net-model && \
    git lfs pull && \
    cd .. && \
    mv XL-Net-model flask-mlZ6/XL-Net-model

# Navigate into the cloned directory
WORKDIR /app/flask-mlZ6

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["python", "app.py"]