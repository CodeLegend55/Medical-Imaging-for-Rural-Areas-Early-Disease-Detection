import unittest
from src.models.resnet50_model import ResNet50Model
from src.models.densenet121_model import DenseNet121Model
from src.models.efficientnet_model import EfficientNetModel

class TestModels(unittest.TestCase):

    def setUp(self):
        self.resnet_model = ResNet50Model()
        self.densenet_model = DenseNet121Model()
        self.efficientnet_model = EfficientNetModel()

    def test_resnet_model_initialization(self):
        self.assertIsNotNone(self.resnet_model)

    def test_densenet_model_initialization(self):
        self.assertIsNotNone(self.densenet_model)

    def test_efficientnet_model_initialization(self):
        self.assertIsNotNone(self.efficientnet_model)

    def test_resnet_model_prediction(self):
        # Assuming the model has a predict method and input shape is (224, 224, 3)
        test_input = np.random.rand(1, 224, 224, 3)
        prediction = self.resnet_model.predict(test_input)
        self.assertEqual(prediction.shape[1], 4)  # Assuming 4 classes

    def test_densenet_model_prediction(self):
        test_input = np.random.rand(1, 224, 224, 3)
        prediction = self.densenet_model.predict(test_input)
        self.assertEqual(prediction.shape[1], 4)

    def test_efficientnet_model_prediction(self):
        test_input = np.random.rand(1, 224, 224, 3)
        prediction = self.efficientnet_model.predict(test_input)
        self.assertEqual(prediction.shape[1], 4)

if __name__ == '__main__':
    unittest.main()