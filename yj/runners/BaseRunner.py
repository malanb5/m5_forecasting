from abc import ABC, abstractmethod, ABCMeta

class BaseRunner(ABC, metaclass=ABCMeta):
    @abstractmethod
    def run(self, actions, cuda):
        pass

    @abstractmethod
    def train(self, cuda):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass