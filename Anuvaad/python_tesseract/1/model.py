import numpy as np
import triton_python_backend_utils as pb_utils
import pytesseract
import os
import urllib.request

class TritonPythonModel:
    def initialize(self, args):
        os.environ["TESSDATA_PREFIX"] = "/home"
        if not os.path.exists("anuvaad_hi.traineddata"):
            urllib.request.urlretrieve("https://anuvaad-pubnet-weights.s3.amazonaws.com/anuvaad_hin.traineddata", "anuvaad_hi.traineddata")  
        if not os.path.exists("anuvaad_ml.traineddata"):
            urllib.request.urlretrieve("https://anuvaad-pubnet-weights.s3.amazonaws.com/anuvaad_mal.traineddata", "anuvaad_ml.traineddata")  

        

    def execute(self, requests):
        responses = []
        for request in requests:
            input_image = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE").as_numpy()
            lang_id = pb_utils.get_input_tensor_by_name(request, "INPUT_LANGUAGE_ID").as_numpy()
            lang_id = lang_id[0].decode("utf-8", "ignore")
            lang_id = "anuvaad_"+lang_id
            generated_text = pytesseract.image_to_string(input_image, config='--psm 7',lang=lang_id) 
            print(generated_text)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "OUTPUT_TEXT", np.array([generated_text.encode()]),
                    )
                ]
            )
            responses.append(inference_response)
        return responses 