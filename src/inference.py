from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from PIL import Image
import requests
from io import BytesIO

from src.vlm.model.loader import load_model, load_tokenizer
from src.vlm.hyparams.parser import get_infer_args, read_args
from qwen_vl_utils import process_vision_info
from utils.logger import logger


@dataclass
class VLMConfig:
    model_args: Dict[str, Any]
    data_args: Dict[str, Any]
    finetuning_args: Dict[str, Any]
    generating_args: Dict[str, Any]

    @classmethod
    def from_args(cls, args: Optional[Dict[str, Any]] = None) -> "VLMConfig":
        args = read_args(args)
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        return cls(model_args, data_args, finetuning_args, generating_args)


class ImageLoader:
    @staticmethod
    def load(source: Union[str, Path]) -> Image.Image:
        """
        Load image from either URL or local path

        Args:
            source: URL or local path to image

        Returns:
            PIL.Image: Loaded image

        Raises:
            ValueError: If source is invalid or image cannot be loaded
        """
        # Convert Path to string if needed
        source = str(source)

        try:
            # Check if source is URL
            if source.startswith(("http://", "https://")):
                response = requests.get(source, timeout=10)
                response.raise_for_status()  # Raise error for bad status codes
                return Image.open(BytesIO(response.content))

            # Treat as local path
            else:
                return Image.open(source)

        except (requests.RequestException, OSError) as e:
            raise ValueError(f"Failed to load image from {source}: {str(e)}")


class MessageBuilder:
    @staticmethod
    def create_ocr_message(image: Image.Image) -> List[Dict[str, Any]]:
        """Create a message template for OCR task"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "請偵測圖片中的所有文字，並標記出它們的位置",
                    },
                ],
            }
        ]


class VLMInference:
    def __init__(self, config: VLMConfig):
        self.config = config
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize tokenizer, processor and model"""
        tokenizer_module = load_tokenizer(self.config.model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.model = load_model(
            self.tokenizer, self.config.model_args, self.config.finetuning_args
        )

    def process_inputs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process inputs for model inference"""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def generate(self, inputs: Dict[str, Any]) -> List[str]:
        """Generate output from processed inputs"""

        skip_special_tokens_flag = self.config.generating_args.__dict__.pop(
            "skip_special_tokens", False
        )
        generated_ids = self.model.generate(
            **inputs, **self.config.generating_args.__dict__
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=skip_special_tokens_flag,
            clean_up_tokenization_spaces=False,
        )

    def infer(self, image: Image.Image) -> List[str]:
        """Run complete inference pipeline"""
        messages = MessageBuilder.create_ocr_message(image)
        inputs = self.process_inputs(messages)
        return self.generate(inputs)


def main(image_file: str = "test.jpeg", args: Optional[Dict[str, Any]] = None) -> None:
    # Initialize configuration
    config = VLMConfig.from_args(args)

    # Initialize inference pipeline
    inference = VLMInference(config)

    # Load image - works with both URL and local path
    image = ImageLoader.load(image_file)

    # Run inference
    output = inference.infer(image)
    logger.info(f"The Inference result: {output}")


if __name__ == "__main__":
    main()
