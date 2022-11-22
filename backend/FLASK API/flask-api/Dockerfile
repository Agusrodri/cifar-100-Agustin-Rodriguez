FROM python:3.10.8

COPY . /src

WORKDIR src

EXPOSE 8000:8000

RUN pip3 install -r requirements.txt

CMD [ "python3", "app.py" ]