from transformerkp.data.base import Dataset


class KGDataset(Dataset):

    def __init__(self, data_args):
        super().__init__()
        self.data_args = data_args
        self._train = None
        self._validation = None
        self._test = None

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test