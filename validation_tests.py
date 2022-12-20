import pytest
import os

from logger import initialize_logger


initialize_logger(debug=True)

# LOGGER=None
# logging.disable(logging.CRITICAL)
# LOGGER.disabled = True


from main import ItemsTreeDS


tree_instance = ItemsTreeDS(regenerate=True)
test_data_folder = "testing_directory"
test_data_folder = os.path.abspath(test_data_folder)

dependent_str = "validation_tests.py::"


def test_1_create_new():
    instance = ItemsTreeDS(regenerate=True)
    instance.do_tree_check()


def test_1_do_tree_check():
    tree_instance.do_tree_check()


def test_1_got_currencies():
    # tree_instance.do_tree_check()
    assert 0 < tree_instance.euro < 300
    assert 0 < tree_instance.usd < 300


def test_1_got_traders():
    assert tree_instance.traders_keys, "No trader keys?"


class Test_Load():
    tree_instance.save(test_data_folder)
    loaded_instance = ItemsTreeDS(load_path=test_data_folder)

    def test_2_load_check(self):
        assert self.loaded_instance._loaded, "Should load"

    # @pytest.mark.dependency(depends=['Test_Load::test_2_load_check'])
    def test_2_multi_load_check(self):
        instance = ItemsTreeDS()
        assert not instance._loaded, "Should not be loaded"
        instance.load(test_data_folder)
        instance.load(test_data_folder)
        instance.load(test_data_folder)
        assert instance._loaded, "Should load"
        instance.do_tree_check()

    # @pytest.mark.dependency(depends=['Test_Load::test_2_new_instance'])
    def test_2_do_tree_check(self):
        self.loaded_instance.do_tree_check()

    # @pytest.mark.dependency(depends=['Test_Load::test_2_load_check'])
    def test_2_load_currency(self):
        # self.loaded_instance.do_tree_check()
        assert 0 < self.loaded_instance.euro < 300
        assert 0 < self.loaded_instance.usd < 300

    # @pytest.mark.dependency(depends=['Test_Load::test_2_new_instance'])
    def test_2_load_trader_keys(self):
        assert self.loaded_instance.traders_keys, "No trader keys?"


class TestQueries:

    def test_1_no_parts(self):
        # tree_instance.
        pass
