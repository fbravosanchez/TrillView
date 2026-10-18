# Python image to use.
FROM python:3.12

# Set the working directory to /app
WORKDIR /app

#ENV APP_HOME /app
#WORKDIR $APP_HOME


# copy requirements file
COPY requirements.txt .

# Install needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the working directory at /app
COPY . .

#sets environment port
ENV PORT=80

# Run app.py when the container launches
CMD exec gunicorn --bind :$PORT --workers 4 --threads 2 --timeout 120 app:server 
