# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install uv, the new package manager
RUN pip install uv

# Copy project configuration and source code
COPY pyproject.toml ./
COPY src/ ./src/
COPY cli.py ./

# Copy essential data files that the application needs
# Ensure this file exists in your project root before building the image
COPY simplified_chars.txt ./

# Install project dependencies using uv
# This will read the pyproject.toml file and install everything needed
RUN uv pip install --system .

# Set the entrypoint to our CLI script
# This allows running commands like `docker run <image> process-training-data ...`
ENTRYPOINT ["python", "cli.py"]
