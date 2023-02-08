FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY docker/requirements.txt .

## Install dependencies
RUN apt-get update && \
  apt-get -qq -y --no-install-recommends install apt-utils netcat-openbsd 

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN [ "python3", "-c", "import nltk; nltk.download('stopwords')" ]
RUN cp -r /root/nltk_data /usr/local/share/nltk_data 

# Copy the current directory contents into the container at /app
COPY ./src/ ./src/

COPY ./dataset ./dataset


COPY ./entrypoint.sh ./entrypoint.sh

RUN chmod +x ./entrypoint.sh

CMD  bash ./entrypoint.sh