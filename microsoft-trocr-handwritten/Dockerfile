ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.12-py3
FROM ${BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8


WORKDIR /home

RUN pip install -r requirements.txt




COPY ./python_tesseract ./models

CMD ["tritonserver", "--model-repository=/models", "--log-verbose=2", "--strict-model-config=false", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002"]
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002