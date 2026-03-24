class ModelPipeline:
      def __init__(self, pipeline, model_type, hyperparameters, time, status, metrics):
            self.pipeline = pipeline
            self.model_type = model_type
            self.hyperparameters = hyperparameters
            self.time = time
            self.status = status
            self.metrics = metrics