# 1. Use an official Python runtime as a parent image
FROM python:3.10.17-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container at /app
COPY requirements.txt requirements_waymo_dataset.txt .

# 4. Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the pip cache, which can reduce image size.
# --trusted-host pypi.python.org: Can be useful if you're behind a proxy or have SSL issues.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r requirements_waymo_dataset.txt --no-deps
RUN apt-get update && apt-get install vim -y

# 5. Copy the rest of the application's code into the container at /app
COPY . .

# 6. Make port 8000 available to the world outside this container (if your app uses it)
EXPOSE 8000

# 7. Define environment variables (optional)
ENV NAME=E2EDriver

# 8. Run app.py when the container launches
# CMD ["python", "src/main.py", "--checkpoint-dir", "/app/checkpoints", "--log-dir", "/app/logs", "--train-dataset-dir", "/mnt"]
CMD ["bash"]