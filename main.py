from src.vlm.hyparams.parser import read_args, get_finetune_args
from src.train import main
import sys
from accelerate.commands.launch import launch_command_parser, launch_command


def do_train():
    training_script = "src/train.py"
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/vlm_config.yaml"
    finetuning_args = get_finetune_args(read_args())
    if finetuning_args.use_accelerate:
        accelerate_config_file = (
            sys.argv[2] if len(sys.argv) > 2 else "config/accelerate.yaml"
        )

        accelerate_args = [
            "--config_file",
            accelerate_config_file,
        ]
        parser = launch_command_parser()
        accelerate_args = accelerate_args + [training_script, config_file]
        args = parser.parse_args(accelerate_args)
        launch_command(args)
    else:
        main()


if __name__ == "__main__":
    do_train()
