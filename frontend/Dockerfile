FROM python:3.10.8

COPY . /src

WORKDIR src

EXPOSE 8001:8001

CMD [ "py", "-m" , "http.server", "8001" ]