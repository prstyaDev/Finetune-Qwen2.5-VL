from qwen_vl_utils.vision_process import (
    fetch_image,
    fetch_video,
)
from abc import ABC, abstractmethod
from utils.logger import logger
from typing import Union, TYPE_CHECKING
from tqdm import tqdm


if TYPE_CHECKING:
    # from src.vlm.hyparams.data_args import DataArguments
    from datasets import (
        Dataset,
        IterableDataset,
    )

NEED_FEATURES = ["image", "video", "text"]


class BaseDataPreprocessor(ABC):
    @abstractmethod
    def preprocess(self):
        pass


class VLMPreprocessor:
    def preprocess(
        self,
        all_datasets: Union["Dataset", "IterableDataset"],
    ) -> Union["Dataset", "IterableDataset"]:
        for feature in NEED_FEATURES:
            if feature not in all_datasets.features.keys():
                new_feature = [None] * len(all_datasets)
                all_datasets = all_datasets.add_column(feature, new_feature)

        return all_datasets

    def process_vision_info(
        self,
        vision_infos: Union["Dataset", "IterableDataset"],
        return_video_kwargs: bool = False,
    ) -> Union["Dataset", "IterableDataset"]:
        """
        Example:

        vision_infos = Dataset([
            {"image": Image.Oject, "video": Video},
            {"image": Image.Oject, "video": Video},
            {"image": Image.Oject, "video": Video}
        ])

        """
        # Read images or videos
        # image_inputs = []
        # video_inputs = []
        # video_sample_fps_list = []
        vision_infos = self.preprocess(vision_infos)

        def process_batch(examples):
            batch_size = len(examples["image"]) if "image" in examples else 0
            image_inputs = [None] * batch_size
            video_inputs = [None] * batch_size

            for i in range(batch_size):
                if "image" in examples:
                    if hasattr(examples["image"][i], "mode"):  # PIL Image has mode attr
                        image_inputs[i] = fetch_image(
                            {"image": examples["image"][i], "text": examples["text"][i]}
                        )
                    else:
                        image_inputs[i] = examples["image"][i]

                if "video" in examples and examples["video"][i] is not None:
                    video_inputs[i] = fetch_video(
                        {"video": examples["video"][i], "text": examples["text"][i]}
                    )
            return {"image_inputs": image_inputs, "video_inputs": video_inputs}

        # for vision_info in tqdm(vision_infos, desc="Process Vision Info"):
        #     if "image" in vision_info or "image_url" in vision_info:
        #         if (
        #             vision_info.get("image", None) is not None
        #             or vision_info.get("image_url", None) is not None
        #         ):
        #             image_inputs.append(fetch_image(vision_info))
        #             video_inputs.append([None])
        #     elif "video" in vision_info and vision_info["video"] is not None:
        #         video_input, video_sample_fps = fetch_video(
        #             vision_info, return_video_sample_fps=True
        #         )
        #         video_sample_fps_list.append(video_sample_fps)
        #         video_inputs.append(video_input)
        #         image_inputs.append([None])
        #     else:
        #         raise ValueError("image, image_url or video should in content.")

        # if len(image_inputs) != 0 and len(video_inputs) != 0:
        #     vision_infos = vision_infos.map(
        #         lambda x, idx: {"image": image_inputs[idx], "video": video_inputs[idx]},
        #         with_indices=True,
        #     )
        # elif len(image_inputs) != 0:
        #     vision_infos = vision_infos.map(
        #         lambda x, idx: {"image": image_inputs[idx]},
        #         with_indices=True,
        #     )
        # else:
        #     vision_infos = vision_infos.map(
        #         lambda x, idx: {"video": video_inputs[idx]},
        #         with_indices=True,
        #     )

        # vision_infos = vision_infos.add_column("image_inputs", image_inputs)
        # vision_infos = vision_infos.add_column("video_inputs", video_inputs)

        # if return_video_kwargs:
        #     vision_infos = vision_infos.add_column("fps", video_sample_fps_list)
        #     return vision_infos
        # logger.debug("Complete VLMPreprocessor")
        # return vision_infos

        # Use batched processing
        vision_infos = vision_infos.map(
            process_batch,
            batched=True,
            batch_size=32,  # Adjust batch size based on available memory
        )

        return vision_infos
