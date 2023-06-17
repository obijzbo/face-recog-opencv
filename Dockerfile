# Base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY data ./data
COPY app.py .
COPY face-mobile.h5 .

# Expose the port for the Streamlit app
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]