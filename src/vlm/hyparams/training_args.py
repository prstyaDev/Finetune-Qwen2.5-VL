import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from transformers import Seq2SeqTrainingArguments
from transformers.training_args import _convert_str_dict
from trl import SFTConfig

from utils.misc import use_ray


@dataclass
class RayArguments:
    r"""
    Arguments pertaining to the Ray training.
    """

    ray_run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The training results will be saved at `saves/ray_run_name`."
        },
    )
    ray_num_workers: int = field(
        default=1,
        metadata={
            "help": "The number of workers for Ray training. Default is 1 worker."
        },
    )
    resources_per_worker: Union[dict, str] = field(
        default_factory=lambda: {"GPU": 1},
        metadata={
            "help": "The resources per worker for Ray training. Default is to use 1 GPU per worker."
        },
    )
    placement_strategy: Literal["SPREAD", "PACK", "STRICT_SPREAD", "STRICT_PACK"] = (
        field(
            default="PACK",
            metadata={
                "help": "The placement strategy for Ray training. Default is PACK."
            },
        )
    )

    def __post_init__(self):
        self.use_ray = use_ray()
        if isinstance(
            self.resources_per_worker, str
        ) and self.resources_per_worker.startswith("{"):
            self.resources_per_worker = _convert_str_dict(
                json.loads(self.resources_per_worker)
            )


@dataclass
class TrainingArguments(RayArguments, Seq2SeqTrainingArguments):
    r"""
    Arguments pertaining to the trainer.
    """

    def __post_init__(self):
        Seq2SeqTrainingArguments.__post_init__(self)
        RayArguments.__post_init__(self)
        # self.__dict__["sortish_sampler"] = None
        # if hasattr(self, "sortish_sampler"):
        #     delattr(self, "sortish_sampler")

        # if "sortish_sampler" in self.__dict__:
        #     del self.__dict__["sortish_sampler"]


# class SFTTrainingArguments(SFTConfig):
#     def __init__(self, **kwargs):
#         valid_params = SFTConfig.__dataclass_fields__.keys()
#         valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
#         breakpoint()
#         super().__init__(**valid_kwargs)

#         for k, v in kwargs.items():
#             if k not in valid_params and hasattr(self, k):
#                 setattr(self, k, v)

#     def __post_init__(self):
#         SFTConfig.__post_init__(self)
