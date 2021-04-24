import unittest

import torch
from parameterized import parameterized

from utils.config import HYPER_PARAMETERS, MODELS


class Config(object):
    def __init__(self, d):
        self.__dict__ = d


def predict(model, sample_data):
    model.eval()
    with torch.no_grad():
        return model(sample_data)


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_data = torch.randn(1, 256, 64)
        cls.config = Config(HYPER_PARAMETERS)

    @parameterized.expand(MODELS.keys())
    def test(self, name):
        model = MODELS[name][1](self.config)
        gender, accent = predict(model, self.sample_data)
        self.assertEqual(gender.shape, torch.Size([1, 3]), "Gender output shape not equal")
        self.assertEqual(accent.shape, torch.Size([1, 16]), "Accent output shape not equal")
        del model


if __name__ == '__main__':
    unittest.main()
