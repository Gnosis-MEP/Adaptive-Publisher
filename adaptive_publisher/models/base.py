class BaseModel():
    def __init__(self):
        pass

    def predict(self, new_image_frame, last_key_frame=None):
        raise NotImplementedError()
        value = 0.5
        return value