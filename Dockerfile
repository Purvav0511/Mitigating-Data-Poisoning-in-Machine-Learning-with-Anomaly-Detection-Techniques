# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data directory into the container
COPY data /app/data

# Define environment variable
ENV NAME World

# Run script when the container launches
CMD ["python", "poison_cp.py"]
