# base on the official image of python 3.10
FROM python:3.10

# Set the working directory in the container
WORKDIR /streamlit

# Copy the requirements file into the container at /tmp
COPY requirements.txt /tmp

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r /tmp/requirements.txt

# Copy the current directory contents into the container at /streamlit
COPY . /streamlit

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
