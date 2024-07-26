# Use the official Python base image
FROM python:3.9

# Create the directory for NLTK data
RUN mkdir -p /root/nltk_data

# Copy the support Vander lexicon file to the directory
COPY nltk_data/vader_lexicon.zip /root/nltk_data/sentiment/

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app.py file to the container
COPY . .

# Expose the port on which the app will run (change the port number if needed)
EXPOSE 5000

# Set the entry point for the container (command to run the app)
CMD ["python", "app.py"]