from datasets import (
    concatenate_datasets,
    interleave_datasets,
)
from utils.logger import logger
from typing import List, Union, Any, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from src.vlm.hyparams.data_args import DataArguments
    from datasets import (
        DatasetDict,
        Dataset,
        IterableDataset,
    )


class BaseDatasetHandler(ABC):
    @abstractmethod
    def rename_columns(self):
        pass


class DatasetHandler(BaseDatasetHandler):
    def rename_columns(
        self,
        dataset: Union["Dataset", "IterableDataset"],
        dataset_columns: dict,
    ) -> Union["Dataset", "IterableDataset"]:
        for new_col, old_col in dataset_columns.items():
            if new_col == old_col:
                continue
            dataset = dataset.rename_column(old_col, new_col)

        return dataset

    def convert2Dataset(
        self,
        all_datasets: List[Union["Dataset", "IterableDataset", List[Dict], Dict]],
        dataset_columns: List[dict],
    ) -> List[Union["Dataset", "IterableDataset"]]:
        convert_datasets = []
        logger.debug("Convert datasets to Dataset or IterableDataset")
        for sub_dataset in all_datasets:
            if isinstance(sub_dataset, Dict):
                convert_datasets.append(Dataset.from_dict(sub_dataset))
            elif isinstance(sub_dataset, List):
                convert_datasets.append(Dataset.from_list(sub_dataset))
            else:
                convert_datasets.append(sub_dataset)

        return [
            self.rename_columns(subdata, columns)
            for columns, subdata in zip(dataset_columns, convert_datasets)
        ]

    def merge_dataset(
        self,
        all_datasets: List[Union["Dataset", "IterableDataset", List[Any]]],
        dataset_columns: List[dict],
        data_args: "DataArguments",
        seed: int,
        shuffle: bool = False,
    ) -> Union["Dataset", "IterableDataset"]:
        """
        Merges multiple datasets to a unified dataset.
        """
        all_datasets = self.convert2Dataset(all_datasets, dataset_columns)

        logger.debug("Merge all datasets")
        if len(all_datasets) == 1:
            return all_datasets[0]

        elif data_args.mix_strategy == "concat":
            if data_args.streaming:
                logger.warning(
                    "The samples between different datasets will not be mixed in streaming mode."
                )
            return (
                concatenate_datasets(all_datasets).shuffle(seed=seed)
                if shuffle
                else concatenate_datasets(all_datasets)
            )

        elif data_args.mix_strategy.startswith("interleave"):
            if not data_args.streaming:
                logger.warning(
                    "We recommend using `mix_strategy=concat` in non-streaming mode."
                )

            return interleave_datasets(
                datasets=all_datasets,
                probabilities=data_args.interleave_probs,
                seed=seed,
                stopping_strategy=(
                    "first_exhausted"
                    if data_args.mix_strategy.endswith("under")
                    else "all_exhausted"
                ),
            )
        else:
            raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")

    def split_dataset(
        self,
        dataset: Union["Dataset", "IterableDataset"],
        data_args: "DataArguments",
        seed: int,
    ) -> "DatasetDict":
        """
        Splits the dataset and returns a dataset dict containing train set and validation set.

        Supports both map dataset and iterable dataset.
        """
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
            val_set = dataset.take(int(data_args.val_size))
            train_set = dataset.skip(int(data_args.val_size))
            return DatasetDict({"train": train_set, "validation": val_set})
        else:
            val_size = (
                int(data_args.val_size)
                if data_args.val_size > 1
                else data_args.val_size
            )
            dataset = dataset.train_test_split(test_size=val_size, seed=seed)
            return DatasetDict(
                {"train": dataset["train"], "validation": dataset["test"]}
            )
