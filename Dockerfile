FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3

# Install Dependencies
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Set Timezone
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# copy all scripts
COPY . /app

WORKDIR /root/src

ENTRYPOINT ["uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0"]