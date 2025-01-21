# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

ENV OPENAI_API_KEY="xxx"
ENV LANGSMITH_TRACING="true"
ENV LANGSMITH_API_KEY="xxx"

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Run the script when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]