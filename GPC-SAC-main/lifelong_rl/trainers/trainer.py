import abc


class Trainer(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, data,w):
        #pass
        return  q



    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}
    def pre_train(self,data):
        pass
    def pre_train2(self,data):
        pass