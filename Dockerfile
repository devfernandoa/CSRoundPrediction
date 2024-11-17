FROM python:3.12-slim
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application, including the data directory
COPY . .

# Make sure the data directory exists in the container
RUN mkdir -p /app/data/models

# Copy specifically the model file
COPY ./data/models/csgo_model.pkl /app/data/models/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
