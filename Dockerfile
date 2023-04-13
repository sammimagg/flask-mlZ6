# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install Git LFS and Git
RUN apt-get update && \
    apt-get install -y git-lfs git && \
    git lfs install

# Set the working directory to /app
WORKDIR /app

# Download the repo
RUN git clone https://github.com/sammimagg/flask-mlZ6.git

# Navigate into the cloned directory
WORKDIR /app/flask-mlZ6

# Pull the large file using Git LFS
RUN git lfs pull

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["python", "main.py"]
