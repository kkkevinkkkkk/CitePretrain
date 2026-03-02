import pandas as pd
import torch
import fire
import os
import numpy as np
import gc

from pipeline import (MyPipeline, pipeline_init)
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from omegaconf import OmegaConf
from tqdm import tqdm
from prompter import Prompter
from utils import read_jsonl, save_jsonl, OPENAI_MODELS, multi_process_map
from eval import Evaluator
from logits_processors import CitationDecodingProcessor, RestrictTokensLogitsProcessor
from generation import load_model_with_citation_generation, CitationGenerationConfig
from vllm.distributed.parallel_state import destroy_model_parallel

def main(
        config_path=None
):
    if config_path is None:
        raise ValueError("config_path is required. Example: --config_path configures/eval_asqa.yml")
    args = OmegaConf.load(config_path)
    print(OmegaConf.to_container(args, resolve=True))
    model = args.model
    model_name = args.model.split("/")[-1] if args.get("model_name", None) is None else args.model_name

    if args.get("model_path", None) is not None:
        model = args.model_path if args.get("use_vllm", False) else (
            AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto"))
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model) if model not in OPENAI_MODELS else None


    eos_token_id = tokenizer.eos_token_id if model not in OPENAI_MODELS else 0

    confidence_method = args.get("confidence_method", "")
    confidence_to_pipeline = {
        "": MyPipeline,
    }

    num_return_sequences = args.get("num_return_sequences", 1)
    postprocess_funcs = args.get("postprocess_funcs", [])

    pipe = pipeline_init(
        task="text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        pipeline_class=confidence_to_pipeline[confidence_method],
        model_name=model_name,
        tokenizer=tokenizer,
        skip_special_tokens=args.get("skip_special_tokens", True),
        postprocess_funcs=postprocess_funcs,
        use_vllm=args.get("use_vllm", False),
    )

    citation_generation_config = None
    if args.get("citation_beam_size", None) is not None:
        pipe.model = load_model_with_citation_generation(pipe.model)
        SOT_TOKEN = "<|reserved_special_token_0|>"
        EOT_TOKEN = "<|reserved_special_token_1|>"
        citation_start_token_id = tokenizer.convert_tokens_to_ids(SOT_TOKEN)
        citation_end_token_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
        citation_generation_config = CitationGenerationConfig(
            citation_beam_size=args.citation_beam_size,
            citation_start_token_id=citation_start_token_id ,
            citation_stop_token_id=citation_end_token_id, )

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data
    eval_data = read_jsonl(args.eval_file)

    # sample part of the data
    if args.sample_size > 0:
        sample_start = args.get("sample_start", 0)
        eval_data = eval_data[sample_start: sample_start + args.sample_size]
        print(f"sample from {sample_start} to {sample_start + args.sample_size}")


    else:
        args.sample_size = len(eval_data)

    prompter_ = Prompter(
        n_shot=args.n_shot,
        n_doc=args.n_doc,
        n_doc_in_demo=args.get("n_doc_in_demo", 0),
        no_doc_in_demo=args.get("no_doc_in_demo", True),
        dataset_name=args.dataset_name,
        model_name=args.model,
        demo_prompt_idx=args.get("demo_prompt_idx", None),
    )


    max_new_tokens = args.get("max_new_tokens", 512)
    prompter_task_type = args.get("task_type", "main")
    logits_processor = None
    if args.get("num_options", 0) > 0:
        allowed_tokens = [chr(ord('A') + i) for i in range(args.num_options)]
        allowed_tokens_preprocessor = RestrictTokensLogitsProcessor(tokenizer, allowed_tokens=allowed_tokens)
        logits_processor = [allowed_tokens_preprocessor]
        max_new_tokens = 1
        prompter_task_type = "multiple_choice"

    if args.get("citations_list_file", None) is not None:
        citations_list = read_jsonl(args.citations_list_file, return_df=False)
        citations_list = [item["title"] for item in citations_list]
        citations_preprocessor = CitationDecodingProcessor(tokenizer, citations_list)
        logits_processor = [citations_preprocessor]


    temperature = args.get("temperature", 0.6)

    def get_model_answer(eval_item):
        # turn df to dict if needed
        if isinstance(eval_item, pd.Series):
            eval_item = eval_item.to_dict()
        text_input = prompter_.generate_text_input(task_type=prompter_task_type,
                                                   eval_item=eval_item,
                                                   faithful_type=args.get("faithful_type", None),)

        eval_item['text_input'] = text_input

        if args.get("answer_file", None) is None:
            sequences = pipe(
                text_input,
                do_sample=True if temperature > 0 and args.get("num_beams", 1) == 1 else False,
                top_k = args.get("top_k", 50),
                top_p=args.get("top_p", 1.0),
                num_beams=args.get("num_beams", 1),
                num_return_sequences=num_return_sequences,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                random_state=args.seed,
                temperature=temperature,
                logits_processor=logits_processor,
                prompter=prompter_,
                eval_item=eval_item,
                citation_generation_config=citation_generation_config,
            )

            eval_item.update(sequences[0])
            other_answers = [item["generated_text"] for item in sequences[1:]]
        else:
            other_answers = None

        eval_item["other_answers"] = other_answers
        answer = eval_item["generated_text"]
        return eval_item


    # run the model for inference, for gpt model, we can use multi-process to speed up
    if args.get("multi_process", False):
        eval_data = multi_process_map(eval_data, get_model_answer, num_proc=64)
    else:
        eval_data_output = []
        for idx, eval_item in tqdm(eval_data.iterrows()):
            if args.get("debug", False):
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)

            eval_data_output.append(get_model_answer(eval_item))
        eval_data = pd.DataFrame(eval_data_output)




    save_dir = os.path.join(args.save_dir, f"results/{args.exp_name}/{args.dataset_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"create {save_dir}")
    save_path = os.path.join(save_dir, f"{model_name}_predictions")


    sample_suffix = f"_{args.sample_start}:{args.sample_start + args.sample_size}" \
        if args.get("sample_start", None) is not None else f"_{args.sample_size}"
    n_shot_suffix = f"_{args.n_shot}-shot"
    temperature_suffix = f"_t:{temperature}" if temperature != 0.6 else ""
    beam_suffix = f"_{args.num_beams}-beams" if args.get("num_beams", 1) != 1 else ""
    save_path = save_path + n_shot_suffix + temperature_suffix + beam_suffix + sample_suffix + ".jsonl"
    if args.get("save_path", None) is not None:
        save_path = args.save_path

    save_jsonl(eval_data, save_path)


    # evaluate the model if needed
    if args.get("do_eval", True):
        if args.get("use_vllm", False):
            destroy_model_parallel()
            del pipe.model.llm_engine.model_executor.driver_worker
            del pipe.model  # Isn't necessary for releasing memory, but why not
            del pipe
            gc.collect()
            torch.cuda.empty_cache()


        prediction_file = save_path
        evaluator = Evaluator(prediction_file)

        scores = evaluator.evaluate()
        print(scores)

        df = evaluator.df
        save_jsonl(df, prediction_file + ".score")

if __name__ == '__main__':
    fire.Fire(main)
