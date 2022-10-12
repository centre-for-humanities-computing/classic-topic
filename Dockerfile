FROM python:3.9-slim-bullseye

RUN pip install gunicorn
RUN pip install "jax[cpu]"
RUN pip install plotly
RUN pip install tweetopic
RUN pip install numpy
RUN pip install pandas
RUN pip install sklearn
RUN pip install dash

COPY src src

EXPOSE 8080

CMD cd src && gunicorn --timeout 0 -b 0.0.0.0:8080 --worker-tmp-dir /dev/shm --workers=2 --threads=4 --worker-class=gthread main:server