import unittest

import env

from muidae.dataset.rating_dataset import RatingDataset
 
class TestMUIDAE(unittest.TestCase):
 
    def test_all_module(self):
        
        self.test_dataset_module()
        self.test_model_module()
        self.test_tool_module()


    def test_dataset_module(self):

        import test_dataset


    def test_model_module(self):

        import test_model


    def test_tool_module(self):

        import test_tool
