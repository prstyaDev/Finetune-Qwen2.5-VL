from src.inference import VLMConfig, VLMInference
from src.vlm.data.loader import _get_merged_dataset
from typing import Optional, Dict, Any
from src.vlm.hyparams.training_args import TrainingArguments
import re
from evaluate import load
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        self.config = VLMConfig.from_args(args)
        self.inference = VLMInference(self.config)

    def load_evaluation_dataset(self):
        data = _get_merged_dataset(
            dataset_names=self.config.data_args.dataset,
            model_args=self.config.model_args,
            data_args=self.config.data_args,
            training_args=TrainingArguments(output_dir=None),
        )
        return data

    @abstractmethod
    def _inference(self, data):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class OCREvaluator(BaseEvaluator):
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        r"""
        \s* - 匹配零個或多個空白字元（包括空格、tab、換行等）
        () 是捕獲組，代表我們要提取的部分
        [^"] 代表匹配除了引號以外的任何字元
        + 代表匹配一次或多次
        """
        super().__init__(args=args)
        self.pattern = r'"text_content":\s*"([^"]+)"'
        self.cer = load("cer")

    def _inference(self, data):
        for subdata in data:
            response = self.inference.infer(subdata["image"])
            response = (
                "".join(re.findall(self.pattern, response[0]))
                if "text_context" in response[0]
                else response[0]
            )

            yield response.replace("<|im_end|>", ""), subdata["text"]

    def evaluate(self):
        data = self.load_evaluation_dataset()
        data = data.select(range(10))
        pred_list = []
        grond_truth_list = []
        result = self._inference(data)

        for predict, truth in tqdm(result, total=len(data)):
            pred_list.append(predict)
            grond_truth_list.append(truth)

        cer_score = self.cer.compute(predictions=pred_list, references=grond_truth_list)
        return cer_score, pred_list, grond_truth_list


if __name__ == "__main__":
    evaluator = OCREvaluator()
    score, pred_list, grond_truth_list = evaluator.evaluate()
    print(score)
