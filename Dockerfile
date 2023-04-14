# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

ENV LISTEN_PORT=5000
EXPOSE 5000

WORKDIR /app

# Install Dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader vader_lexicon
RUN python -m nltk.downloader omw-1.4

# Set Timezone
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# copy all scripts
COPY /root /app/root

ENTRYPOINT ["uvicorn", "--app-dir=./root/src", "main:app", "--port", "5000", "--host", "0.0.0.0"]