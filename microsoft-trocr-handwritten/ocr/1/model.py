import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TritonPythonModel:
    def initialize(self, args):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
        

    def execute(self, requests):
        responses = []
        for request in requests:
            input_image = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE").as_numpy()
            pixel_values = self.processor(images=input_image, return_tensors="pt").pixel_values

            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]   
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