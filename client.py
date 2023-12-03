import argparse
import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *

def main(model_name):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    url = "https://anuvaad-raw-datasets.s3-us-west-2.amazonaws.com/anuvaad_ocr_malayalam.jpg"
    language = "ml"
    
    image = np.asarray(Image.open(requests.get(url, stream=True).raw))
    input_language_id = np.array([language], dtype="object")

    # Set Inputs
    input_tensors = [httpclient.InferInput("INPUT_IMAGE", image.shape, datatype=np_to_triton_dtype(image.dtype)), httpclient.InferInput("INPUT_LANGUAGE_ID", input_language_id.shape, datatype=np_to_triton_dtype(input_language_id.dtype))]
    input_tensors[0].set_data_from_numpy(image)
    input_tensors[1].set_data_from_numpy(input_language_id)

    # Set outputs
    outputs = [httpclient.InferRequestedOutput("OUTPUT_TEXT")]

    # Query
    query_response = client.infer(
        model_name=model_name, inputs=input_tensors, outputs=outputs
    )
 
    # Output
    output = query_response.as_numpy("OUTPUT_TEXT")
    print(output[0].decode('UTF-8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #specify deployed model name here eg: python_trocr
    parser.add_argument(
        "--model_name", default="python_tesseract"
    )
    args = parser.parse_args()
    main(args.model_name)