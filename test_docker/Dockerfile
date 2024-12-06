# Use an official Python runtime as a parent image
FROM python:3.7
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install pip
RUN apt-get update && apt-get install -y python3-pip && pip install --upgrade pip

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Install recbole version 1.0.1
# RUN pip install recbole==1.0.1 --default-timeout=100 --retries=10

# Install torch version >=1.7.0
# RUN pip install torch>=1.7.0

# Make port 80 available to the world outside this container
EXPOSE 80

# # Define environment variable
# ENV NAME World

# Run app.py when the container launches
CMD ["python", "/app/test.py"]