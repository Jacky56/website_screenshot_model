FROM python:3.9.6-buster

RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get update -y
RUN apt-get install -y google-chrome-stable

RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/bin/

COPY requirements.txt /.
WORKDIR /.
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY model/test_640_360_3-2.h5 /app/model/

WORKDIR /app/

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app