class DataTransformer:
    def normalize(self, value, min, max):
        return (value - min) / (max - min)

    def denormalize(self, value, min, max):
        return (value * (max - min)) + min
