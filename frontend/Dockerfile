FROM python:3.10.8

COPY . /src

WORKDIR src

EXPOSE 8001:8001

CMD [ "python3", "-m" , "http.server", "8001" ]