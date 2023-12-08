# Triton_inference
Repo contains code to deploy and infer several open sourced indic models utilizing nvidia triton server

**Running the server**

```shell
sudo docker build -f <model-dir>/Dockerfile -t triton_tesseract .
sudo docker run -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models  triton_tesseract
```


**Running the inference**

```shell
sudo docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.12-py3-sdk bash
python3 client.py --model_name ocr
```
