import torch

from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import Qwen2_5_VLProcessor


# https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py
# https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-finetune-qwen2-5-vl-for-json-data-extraction.ipynb
def create_message_template(batch):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": batch["image"]},
                {"type": "text", "text": "請列出圖片中的文字"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": batch["text"],
                }
            ],
        },
    ]


def find_assistant_content_sublist_indexes(label):
    """
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, \
                    {'type': 'text', 'text': '描述一下这个图片'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': \
                    '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，\
                        坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广\
                            阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>\
            <|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场\
                景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋\
                    和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    """
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(label) - 2):
        # Check if the current and next elements form the start sequence
        if label[i] == 151644 and label[i + 1] == 77091 and label[i + 2] == 198:
            start_indexes.append(i + 3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i + 3, len(label) - 1):
                if label[j] == 151645 and label[j + 1] == 198:
                    end_indexes.append(j + 2)
                    # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model \can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


@dataclass
class DataCollatorForQwenVL:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        text, image_inputs, video_inputs = map(
            list,
            zip(
                *[
                    (
                        self.processor.apply_chat_template(
                            create_message_template(batch),
                            tokenize=False,
                            add_generation_prompt=False,
                        ),
                        # batch["text"],
                        batch["image_inputs"],
                        batch["video_inputs"],
                    )
                    for batch in features
                ]
            ),
        )
        batch = self.processor(
            text=text,
            images=image_inputs if image_inputs[0] is not None else None,
            videos=video_inputs if video_inputs[0] is not None else None,
            padding=True,
            return_tensors="pt",
        )
        # labels = batch["input_ids"].clone()
        # if self.processor.tokenizer.pad_token_id is not None:
        #     labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # batch["labels"] = labels

        labels_list = batch["input_ids"].clone()  # .tolist()
        labels_list[labels_list == self.processor.tokenizer.pad_token_id] = -100

        if isinstance(self.processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [
                self.processor.tokenizer.convert_tokens_to_ids(
                    self.processor.image_token
                )
            ]

        # labels_list = []
        # for ids_list in input_ids_lists:
        # label_ids = [-100] * len(ids_list)
        # for begin_end_indexs in find_assistant_content_sublist_indexes(
        #     ids_list, image_tokens
        # ):
        #     label_ids[begin_end_indexs[0] : begin_end_indexs[1]] = ids_list[
        #         begin_end_indexs[0] : begin_end_indexs[1]
        #     ]
        for image_token_id in image_tokens:
            labels_list[labels_list == image_token_id] = -100

        batch["labels"] = torch.tensor(labels_list, dtype=torch.int64)

        return batch
