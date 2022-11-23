FROM python:3.10.8

COPY . /src

WORKDIR src

EXPOSE 8002:8002

RUN pip install -r requirements.txt

CMD [ "python3", "app.py" ]