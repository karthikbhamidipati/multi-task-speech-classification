import unittest

import torch

from models import run_device
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.simple_cnn import simple_cnn
from utils.config import HYPER_PARAMETERS


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

    def test_simple_cnn(self):
        model = simple_cnn(self.config)
        gender, accent = predict(model, self.sample_data)
        self.assertEqual(gender.shape, torch.Size([1, 3]), "Gender output shape not equal")
        self.assertEqual(accent.shape, torch.Size([1, 16]), "Accent output shape not equal")
        del model

    def test_resnet18(self):
        model = resnet18(self.config)
        gender, accent = predict(model, self.sample_data)
        self.assertEqual(gender.shape, torch.Size([1, 3]), "Gender output shape not equal")
        self.assertEqual(accent.shape, torch.Size([1, 16]), "Accent output shape not equal")
        del model

    def test_resnet34(self):
        model = resnet34(self.config)
        gender, accent = predict(model, self.sample_data)
        self.assertEqual(gender.shape, torch.Size([1, 3]), "Gender output shape not equal")
        self.assertEqual(accent.shape, torch.Size([1, 16]), "Accent output shape not equal")
        del model

    def test_resnet50(self):
        model = resnet50(self.config)
        gender, accent = predict(model, self.sample_data)
        self.assertEqual(gender.shape, torch.Size([1, 3]), "Gender output shape not equal")
        self.assertEqual(accent.shape, torch.Size([1, 16]), "Accent output shape not equal")
        del model

    def test_resnet101(self):
        model = resnet101(self.config)
        gender, accent = predict(model, self.sample_data)
        self.assertEqual(gender.shape, torch.Size([1, 3]), "Gender output shape not equal")
        self.assertEqual(accent.shape, torch.Size([1, 16]), "Accent output shape not equal")
        del model


if __name__ == '__main__':
    unittest.main()
