FROM python:3.12-alpine
WORKDIR /site
COPY index.html .
EXPOSE 8080
CMD ["python3", "-m", "http.server", "8080"]
