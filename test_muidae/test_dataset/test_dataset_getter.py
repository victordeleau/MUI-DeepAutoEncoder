import unittest
import os, sys

from muidae.dataset.dataset_getter import DatasetGetter
from muidae.dataset.rating_dataset import RatingDataset

"""
    Test class associated with the DatasetGetter class
"""
class TestDatasetGetter(unittest.TestCase):

    dataset_getter = None

    dataset_name = ['100k', '1m', '10m', '20m', '26m', 'serendipity', '100k_old']

    def initialize(self):

        def decorator(method):

            dataset_getter = DatasetGetter()
            method()

        return decorator

    
    class test_load_local_dataset_object(unittest.TestCase):

        pass


    class test_export_dataset_object_to_disk(unittest.TestCase):

        pass


    class test_local_dataset_found(unittest.TestCase):

        pass


    class test_get_available_dataset(unittest.TestCase):

        pass


    class test_download_dataset(unittest.TestCase):

        pass


    class test_get_available_dataset(unittest.TestCase):

        pass


    class test_save_dataset_to_disk(unittest.TestCase):

        pass


    @initialize
    class test_get(unittest.TestCase):


        def test_should_return_100k_dataset_by_default(self):

            dataset = dataset_getter.get()

            self.assertEqual( dataset.name, "100k" )


        def test_should_not_export_dataset_object_to_disk_by_default(self):

            dataset = dataset_getter.get()

            self.assertFalse( os.path.exists( "data/pickle/100k.dump" ) )


        def test_should_not_raise_exception_for_provided_dataset_name(self):

            for name in dataset_name:
                try:
                    dataset_getter.get( name )
                except:
                    self.fail("Dataset not found exception raise for a dataset name that should be supported.")


        def test_should_export_dataset_object_to_disk_if_requested(self):

            dataset = dataset_getter.get( export_to_disk=True )

            self.assertTrue( os.path.exists( "data/pickle/100k.dump" ) )


        def test_should_return_object_of_type_RatingDataset(self):

            dataset = dataset_getter.get()

            self.assertTrue( isinstance(dataset, RatingDataset) )
