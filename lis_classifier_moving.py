import numpy as np
import tensorflow as tf


class LISClassifierMoving(object):
    def __init__(
        self,
        model_path='model/movexported.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.resize_tensor_input(0,(6,1,6*20))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array(landmark_list, dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        
        result = self.interpreter.get_tensor(output_details_tensor_index)
    
        max_result = max(result[0])

        result_index = np.argmax(np.squeeze(result[0]))%4

        return result_index, max_result 
