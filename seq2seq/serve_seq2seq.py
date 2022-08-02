# Set up logging
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

from typing import Optional, Dict
from dataclasses import dataclass, field
from pydantic import BaseModel
import os
from contextlib import nullcontext
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, HTTPException
from uvicorn import run
from sqlite3 import Connection, connect, OperationalError
from seq2seq.utils.pipeline import Text2SQLGenerationPipeline, Text2SQLInput, get_schema
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments
import pickle


PATH_TO_CACHED_DATA = os.path.dirname(os.path.realpath(__file__)) + "/tmp_cache.bin"
INDEX_CURRENT_HOST = 0


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


def main():
    global INDEX_CURRENT_HOST
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

    # Initialize config
    config = AutoConfig.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        use_fast=True,
    )

    # Initialize Picard if necessary
    with PicardLauncher() if picard_args.launch_picard else nullcontext(None):
        # Get Picard model class wrapper
        if picard_args.use_picard:
            model_cls_wrapper = lambda model_cls: with_picard(
                model_cls=model_cls, picard_args=picard_args, tokenizer=tokenizer
            )
        else:
            model_cls_wrapper = lambda model_cls: model_cls

        # Initialize model
        model = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
            backend_args.model_path,
            config=config,
            cache_dir=backend_args.cache_dir,
        )

        def gen_wrapper(func):
            def f(*args, **kwargs):
                global INDEX_CURRENT_HOST
                kwargs["output_scores"] = True
                kwargs["return_dict_in_generate"] = True
                output = func(*args, **kwargs)
                sequences_scores = output.sequences_scores.cpu()
                l_of_scores_cpu = []
                for score in output.scores:
                    l_of_scores_cpu.append(score.cpu())
                cache = load_data(PATH_TO_CACHED_DATA)
                cache.append(
                    {
                        "INDEX_CURRENT_HOST": INDEX_CURRENT_HOST,
                        "sequences_scores": sequences_scores,
                        "scores": l_of_scores_cpu,
                    }
                )
                store_data(PATH_TO_CACHED_DATA, cache)
                INDEX_CURRENT_HOST += 1
                return output.sequences

            return f

        model.generate = gen_wrapper(model.generate)

        # Initalize generation pipeline
        pipe = Text2SQLGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            db_path=backend_args.db_path,
            prefix=data_training_args.source_prefix,
            normalize_query=data_training_args.normalize_query,
            schema_serialization_type=data_training_args.schema_serialization_type,
            schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
            device=backend_args.device,
        )

        # Initialize REST API
        app = FastAPI()

        class AskResponse(BaseModel):
            query: str
            execution_results: list

        def response(query: str, conn: Connection) -> AskResponse:
            try:
                return AskResponse(query=query, execution_results=conn.execute(query).fetchall())
            except OperationalError as e:
                return AskResponse(query=query, execution_results=[])

        @app.get("/ask/{db_id}/{question}")
        def ask(db_id: str, question: str):
            try:
                outputs = pipe(
                    inputs=Text2SQLInput(utterance=question, db_id=db_id),
                    num_return_sequences=data_training_args.num_return_sequences,
                )
                print("outputs serve_seq2seq", outputs)
            except OperationalError as e:
                raise HTTPException(status_code=404, detail=e.args[0])
            try:
                conn = connect(backend_args.db_path + "/" + db_id + "/" + db_id + ".sqlite")
                picard_result = [response(query=output["generated_text"], conn=conn) for output in outputs]
                return picard_result
            finally:
                conn.close()

        # Run app
        run(app=app, host=backend_args.host, port=backend_args.port)


if __name__ == "__main__":
    main()
