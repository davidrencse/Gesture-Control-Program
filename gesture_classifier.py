import numpy as np
import tensorflow as tf

class KeyPointClassifier:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, feature_vec: np.ndarray) -> int:
        """
        feature_vec: (42,) float32
        returns: predicted class index (int)
        """
        x = feature_vec.reshape(1, -1).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        return int(np.argmax(out))
