import random
import numpy as np
from transformers import AutoTokenizer, LlamaTokenizer
from typing import List, Dict
from tqdm import tqdm
from datasets import Dataset
from .utils import read_jsonl
from pandarallel import pandarallel


def write_to_memmap(dset: Dataset, filename: str):
    dtype = np.int32
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = min(1024, len(dset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
        arr.flush()

def sample_paragraphs(paragraphs, start_paragraph_id, end_paragraph_id, min_len=256, max_paragraph_num=3):
    # if the paragraph is not long enough, sample the next paragraph or the previous paragraph randomly
    sampled_paragraphs = paragraphs[start_paragraph_id: end_paragraph_id+1]
    sampled_text = "\n".join(sampled_paragraphs)
    current_paragraph_num = end_paragraph_id - start_paragraph_id + 1
    while len(sampled_text.split(" ")) < min_len and current_paragraph_num < max_paragraph_num:
        if start_paragraph_id > 0 and end_paragraph_id < len(paragraphs) - 1:
            if random.choice([True, False]):
                start_paragraph_id -= 1
            else:
                end_paragraph_id += 1
        elif start_paragraph_id > 0:
            start_paragraph_id -= 1
        elif end_paragraph_id < len(paragraphs) - 1:
            end_paragraph_id += 1
        else:
            break
        sampled_paragraphs = paragraphs[start_paragraph_id: end_paragraph_id+1]
        sampled_text = "\n".join(sampled_paragraphs)
        current_paragraph_num = end_paragraph_id - start_paragraph_id + 1
    return sampled_text.strip()

def split_doc_into_chunks(doc, tokenizer, max_len=1200, sep="\n\n"):
    sub_sep = "\n" if sep == "\n\n" else "." if sep == "\n" else " "
    paragraphs = doc.split(sep)
    paragraphs_with_sep = []
    for p in paragraphs:
        p_len = len(tokenizer(p)["input_ids"])
        if p_len > max_len and sep != sub_sep:
            sub_paragraphs = split_doc_into_chunks(p, tokenizer, max_len=max_len, sep=sub_sep)
            paragraphs_with_sep.extend(sub_paragraphs)
        else:
            paragraphs_with_sep.append({"text": p, "sep": sep, "len": p_len})
    return paragraphs_with_sep


def chunk_raw_doc(row, tokenizer, max_len=1200, modify_title=False):
    # raw_doc_len = len(tokenizer(row['raw_doc']['text'])['input_ids'])
    raw_doc = row['raw_doc']['text']
    chunks_with_sep = split_doc_into_chunks(raw_doc, tokenizer, max_len=max_len)
    chunks = []
    current_chunk = chunks_with_sep[0]
    doc_id = row["raw_doc"]["id"] if "id" in row["raw_doc"] else row["raw_doc"]["title"]
    chunk_id = 0
    for i in range(1, len(chunks_with_sep)):
        chunk = chunks_with_sep[i]
        if current_chunk["len"] + chunk["len"] < max_len:
            current_chunk["text"] += chunk["sep"] + chunk["text"]
            current_chunk["len"] += chunk["len"]
        else:
            title = f"{row['raw_doc']['title']}_Part-{chunk_id}" if modify_title else f"{row['raw_doc']['title']}"
            chunks.append({"text": current_chunk["text"], "title": title, "id": doc_id, "len": current_chunk["len"], "chunk_id": chunk_id})

            current_chunk = chunk
            chunk_id += 1


    title = f"{row['raw_doc']['title']}_Part-{chunk_id}" if modify_title else f"{row['raw_doc']['title']}"
    chunks.append({"text": current_chunk["text"], "title": title, "id": doc_id, "len": current_chunk["len"], "chunk_id": chunk_id})


    return chunks


def sample_multi_granular_chunks(row):
    raw_chunk = row["raw_doc"]['text']
    sampled_texts = []
    # sample one paragraph
    paragraphs = raw_chunk.split("\n")
    paragraphs = [p for p in paragraphs if len(p.strip().split(" ")) > 30]
    for i in range(3):
        if len(paragraphs) == 0:
            break
        sampled_text = np.random.choice(paragraphs)
        sampled_texts.append(sampled_text.strip())

    sentences = raw_chunk.split(".")
    sentences = [s for s in sentences if len(s.strip().split(" ")) > 20]

    # random select 3 sets of 1 sentences
    for i in range(3):
        if len(sentences) == 0:
            break
        sampled_texts.append(np.random.choice(sentences) + ".")

    # random sample 1/3 of the chunk
    n_sentences = len(sentences)

    for i in range(3):
        if len(sentences) == 0:
            break
        start_sentence_idx = np.random.randint(0, len(sentences) - n_sentences // 3)
        end_sentence_idx = start_sentence_idx + n_sentences // 3
        sampled_text = ". ".join(sentences[start_sentence_idx:end_sentence_idx]) + "."
        sampled_texts.append(sampled_text.strip())

    chunks = [row["raw_doc"]]
    for i, sampled_text in enumerate(sampled_texts):
        chunks.append({"text": sampled_text, "title": row["raw_doc"]["title"], "id": row["raw_doc"]["id"] + f"-{i}"})

    return chunks

def generate_pretrain_doc(row, tokenizer, reverse_order=True, add_pretrain_special_tokens=True, add_doc_special_tokens=False, title_special_tokens="reserved", no_title=False):
    SOT_TOKEN = "<|reserved_special_token_0|>"
    EOT_TOKEN = "<|reserved_special_token_1|>"
    SOD_TOKEN = "<|reserved_special_token_2|>"
    EOD_TOKEN = "<|reserved_special_token_3|>"
    if no_title:
        text = row['raw_doc']['text']
        if add_pretrain_special_tokens:
            # text = tokenizer.bos_token + text + tokenizer.eos_token
            text = text + tokenizer.eos_token
        return text
    if title_special_tokens == "html":
        # SOT_TOKEN = "<cite>"
        # EOT_TOKEN = "</cite>"
        SOT_TOKEN = "<source>"
        EOT_TOKEN = "</source>"
    elif title_special_tokens == "angle_brackets":
        SOT_TOKEN = "<"
        EOT_TOKEN = ">"

    title = f"{SOT_TOKEN}{row['raw_doc']['title']}{EOT_TOKEN}"
    if add_doc_special_tokens:
        if reverse_order:
            template = f"{SOD_TOKEN}[doc][title]{EOD_TOKEN}"
        else:
            template = f"{SOD_TOKEN}[title][doc]{EOD_TOKEN}"
    else:
        if reverse_order:
            template = f"[doc][title]"
        else:
            template = f"[title][doc]"

    text = template.replace("[doc]", row['raw_doc']['text']).replace("[title]", title)
    if add_pretrain_special_tokens:
        # text = tokenizer.bos_token + text + tokenizer.eos_token
        text = text + tokenizer.eos_token
    return text





def tokenize_and_save_data(data_path, model_name="meta-llama/Llama-3.2-1B", nb_workers=32):
    pandarallel.initialize(progress_bar=False, nb_workers=nb_workers)
    def _process(example: Dict, tokenizer: LlamaTokenizer, text_key: str) -> Dict:
        """
        Tokenize the text and return the tokenized text
        """
        ids = tokenizer.encode(example[text_key])  # add_special_tokens=True to add BOS token
        # if eos is not in the ids, add it
        if tokenizer.eos_token_id not in ids:
            ids.append(tokenizer.eos_token_id)
        return dict(ids=ids, len=len(ids))

    df = read_jsonl(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    chunk_size = 500000
    num_chunks = int(np.ceil(len(df) / chunk_size))

    token_lists = []  # collects results

    for i in tqdm(range(num_chunks)):
        chunk = df.iloc[i * chunk_size: (i + 1) * chunk_size]

        # Each worker only sees `chunk`, not the full DataFrame
        processed = chunk.parallel_apply(lambda row: _process(row, tokenizer, "text"), axis=1)
        token_lists.extend(processed.tolist())  # free memory early
        del processed, chunk  # encourage GC

    tokenized_ids = Dataset.from_list(token_lists)

    # tokenized_ids = df.parallel_apply(lambda x: _process(x, tokenizer, "text"), axis=1)
    # tokenized_ids = Dataset.from_list(tokenized_ids.tolist())
    save_path = data_path.replace(".jsonl", ".bin")
    print(save_path)
    write_to_memmap(tokenized_ids, save_path)

