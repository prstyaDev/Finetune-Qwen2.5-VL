import pytest
from src.vlm.data.data_utils import DatasetHandler
from datasets import load_dataset, Dataset
from src.vlm.hyparams.data_args import DataArguments
from src.vlm.data.preprocess import VLMPreprocessor
from typing import List


def test_data_handler():
    data = load_dataset("priyank-m/chinese_text_recognition")
    result = DatasetHandler().merge_dataset(
        [data["test"]], [{"image": "image2", "text": "text2"}], DataArguments
    )
    assert isinstance(result, Dataset)


def test_preprocessor():
    data = load_dataset("priyank-m/chinese_text_recognition")
    result = DatasetHandler().merge_dataset(
        [data["test"]], [{"image": "image", "text": "text"}], DataArguments
    )

    result = VLMPreprocessor().process_vision_info(result)
    assert isinstance(result, Dataset)
