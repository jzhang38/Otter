import argparse
import os

import torch
from tqdm import tqdm

import sys

from flamingo.configuration_flamingo import FlamingoConfig
from flamingo.modeling_flamingo import FlamingoForConditionalGeneration

# from .configuration_flamingo import FlamingoConfig
# from .modeling_flamingo import FlamingoForConditionalGeneration

parser = argparse.ArgumentParser(description="Convert Vicuna model")
parser.add_argument("--model_choice", type=str, default="13B", help="Choose either '7B' or '13B'")
parser.add_argument("--save_root_dir", type=str, default="/home/luodian/projects/checkpoints")
args = parser.parse_args()

# os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_choice = args.model_choice
save_root_dir = args.save_root_dir

# prepare vicuna model at first
# you can visit https://huggingface.co/lmsys/Llama-2-33b-chat-hf to download 7B and 30B instruct checkpoints.
if model_choice == "7B":
    config_file = "./flamingo/flamingo-llama2-chat-7B.json"
    state_dict_files = [
        f"outputs/pytorch_model-00001-of-00002.bin",
        f"outputs/pytorch_model-00002-of-00002.bin",
    ]
    save_path = f"{save_root_dir}/flamingo-llama2-chat-7B-init"
elif model_choice == "13B":
    config_file = "./flamingo/flamingo-llama2-chat-13B.json"
    state_dict_files = [
        f"outputs/pytorch_model-00001-of-00003.bin",
        f"outputs/pytorch_model-00002-of-00003.bin",
        f"outputs/pytorch_model-00003-of-00003.bin",
    ]
    save_path = f"{save_root_dir}/flamingo-llama2-chat-13B-init"
else:
    raise ValueError("Invalid model_choice. Choose either '13B' or '7B'.")

config = FlamingoConfig.from_json_file(config_file)
config.vision_config.hidden_size = 1024
import pdb; pdb.set_trace()

model = FlamingoForConditionalGeneration(config=config)

# load flamingo's vision encoder from last checkpoint.
# you can visit https://huggingface.co/luodian/openflamingo-9b-hf/tree/main to download the checkpoint.
# AZP = "os.environ["AZP"]"
import timm
import pdb; pdb.set_trace()
model.vision_encoder = timm.create_model("eva02_large_patch14_clip_224.merged2b", pretrained=True)

# Loading vicuna weights
state_dict = {}
for file in tqdm(state_dict_files, desc="Loading state dict"):
    state_dict_part = torch.load(file, map_location="cpu")
    state_dict.update(state_dict_part)

save_state_dict_1 = {}
for key in state_dict:
    if ".layers." in key:
        _, _, layer_num, *remain_names = key.split(".")
        target_key = f"model.layers.{layer_num}.decoder_layer.{'.'.join(remain_names)}"
    else:
        target_key = key
    save_state_dict_1[f"{target_key}"] = state_dict[key]

# Reshape the token embedding to 50280 for compatible
model.lang_encoder.resize_token_embeddings(32000)

load_msg = model.lang_encoder.load_state_dict(
    save_state_dict_1,
    False,
)
# Reshape the token embedding to 32002 for compatible
model.lang_encoder.resize_token_embeddings(32002)
# print incompatible keys
print(load_msg[1])


print(f"Saving model to {save_path}...")
model.save_pretrained(save_path, max_shard_size="10GB")
