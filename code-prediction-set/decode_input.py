import pdb

# Set up logging
import sys
import logging
# import ipdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

from typing import Optional, Dict
from dataclasses import dataclass, field
import os
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from seq2seq.utils.pipeline import Text2SQLGenerationPipeline, Text2SQLInput, get_schema
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments
import pickle
import torch


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_data(fname):
    if not os.path.exists(fname):
        return []
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


@dataclass
class BackendArguments:
    """
    Arguments pertaining to model serving.
    """

    model_path: str = field(
        default="tscholak/cxmefzzi",
        metadata={"help": "Path to pretrained model"},
    )
    cache_dir: Optional[str] = field(
        default="/tmp",
        metadata={"help": "Where to cache pretrained models and data"},
    )
    db_path: str = field(
        default="database",
        metadata={"help": "Where to to find the sqlite files"},
    )
    host: str = field(default="0.0.0.0", metadata={"help": "Bind socket to this host"})
    port: int = field(default=8000, metadata={"help": "Bind socket to this port"})
    device: int = field(
        default=0,
        metadata={
            "help": "Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU. A non-negative value will run the model on the corresponding CUDA device id."
        },
    )


def generate_txt(l_output_ids, tokenizer):
    generated_text = (
        tokenizer.decode(
            l_output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        .split("|", 1)[-1]
        .strip()
    )
    return generated_text


def main():
    # See all possible arguments by passing the --help flag to this program.
    parser = HfArgumentParser((PicardArguments, BackendArguments, DataTrainingArguments))
    picard_args: PicardArguments
    backend_args: BackendArguments
    data_training_args: DataTrainingArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        picard_args, backend_args, data_training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        picard_args, backend_args, data_training_args = parser.parse_args_into_dataclasses()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        use_fast=True,
    )

    # load data
    data = load_data("/cache/token_probs_beam_size_16_num_pred_8_temp_scaling_10.0.pkl")
    # ipdb.set_trace()
    print("len data", len(data))
    data_with_token = []
    for i in range(len(data)):
        l_prob_token_decode = []
        for j in range(len(data[i]["token_probs"])):
            token_val = data[i]["token_probs"][j]["token"].item()
            l_prob_token_decode.append(
                {
                    "token_encoded": token_val,
                    "token_decoded": generate_txt([token_val], tokenizer),
                    "p": data[i]["token_probs"][j]["p"],
                }
            )
        curr_dict = {
            "indx": data[i]["indx"],
            "token_probs": l_prob_token_decode,
            "entire_query": generate_txt([j["token"].item() for j in data[i]["token_probs"]], tokenizer),
        }
        data_with_token.append(curr_dict)
    print(data_with_token[0])
    store_data("/app/prob_with_token_temp_scaling_10.pkl", data_with_token)


if __name__ == "__main__":
    main()
