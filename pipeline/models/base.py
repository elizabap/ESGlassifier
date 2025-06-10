from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass
