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


    def initialize_dataset_loader(self):

        def decorator(method):

            dataset_getter = DatasetGetter()
            method()

        return decorator


    def initialize_dummy_dataset(self):

        def decorator(method):

            dataset = RatingDataset()
            method()

    
    @initialize_dataset_loader
    class test_download_dataset(unittest.TestCase):

        def test_should_download_100k_by_default(self):

            pass

        def test_should_download_in_local_folder_data_by_default(self):

            pass

        def test_should_write_a_zipfile(self):

            pass


    @initialize_dataset_loader
    class test_get_available_dataset(unittest.TestCase):

        def test_should_return_list_that_match_attribute(self):

            pass


    @initialize_dataset_loader
    class test_local_dataset_found(unittest.TestCase):

        def should_search_100k_by_default(self):

            pass

        def should_search_in_local_data_folder_by_default(self):

            pass

        def test_should_be_true_if_dataset_exist(self):

            pass

        def test_should_return_false_if_dataset_doesnt_exist(self):

            pass


    @initialize_dataset_loader
    class test_load_local_dataset(unittest.TestCase):

        def test_should_load_100k_by_default(self):

            self.assertEqual( dataset.name, "100k" )


        def test_should_search_in_local_data_folder_by_default(self):

            os.makedirs("./nodata/")
            not_found_in_wrong_folder = True
            found_in_correct_folder = True

            try:
                dataset_getter.load_local_dataset(dataset_location="nodata/")
            except Exception as e:
                if "Local dataset not found in folder" in e.message:
                    not_found_in_wrong_folder = True

            try:
                dataset_getter.load_local_dataset(dataset_location="data/")
            except Exception as e:
                found_in_correct_folder = False

            os.removedirs("./nodata/")
            self.assertTrue( not_found_in_wrong_folder and found_in_correct_folder )


        @initialize_dummy_dataset
        def test_should_use_item_view_by_default(self):

            self.assertTrue( dataset.get_view() == "item_view" )


        def test_should_raise_if_file_not_zip(self):

            with open("data/100k.zip", "wb") as f:
                f.write()

            try:
                dataset = dataset_getter.load_local_dataset()
                assert False
            except Exception as e:
                self.assertTrue( e.message == "File is not a Zip file." )

            os.remove("data/100k.zip")


        def test_should_raise_if_no_data_file_found_in_zip_file(self):

            # create a fake zip dataset and assert exception
            pass


        def test_should_raise_if_data_file_is_not_a_zipfile(self):

            pass

        def test_should_raise_if_provided_column_name_are_not_found(self):

            pass

        def test_should_raise_if_provided_column_to_rename_are_not_found(self):

            pass


    @initialize_dataset_loader
    @initialize_dummy_dataset
    class test_get_dataset_loader(unittest.TestCase):

        def test_should_raise_if_dataset_is_not_rating_dataset_object(self):

            try:
                dataset_getter.get_dataset_loader({})
                assert False
            except Exception as e:
                self.assertTrue(e.message == "Provided dataset is not a RatingDataset object")


        def test_should_raise_if_redux_and_shuffle_are_set_together(self):

            try:
                dataset_getter.get_dataset_loader(dataset, redux=0.8, shuffle=True)
                assert False
            except Exception as e:
                self.assertTrue(e.message == "redux is mutually exclusive with shuffle.")


        def test_should_return_dataset_loader_object(self):

            self.assertTrue( isinstance( dataset_getter.get_dataset_loader(dataset), RatingDataset ) )









    @initialize_dataset_loader
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
