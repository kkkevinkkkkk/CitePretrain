from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

def get_openai_completion(prompt, model_name="gpt-4o-mini", temperature=1.0, system_prompt=None):
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(model=model_name,
                                              messages=messages,
                                              temperature=temperature,
                                              logprobs=True)
    output = response.choices[0].message.content
    return output

class NLIModel:
    def __init__(self, model_name="google/t5_11b_trueteacher_and_anli"
):
        # output hidden states
        # model_name = "google/t5_xxl_true_nli_mixture"
        self.openai = True if "gpt" in model_name else False
        self.model = None
        self.tokenizer = None
        if not self.openai:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                               torch_dtype=torch.bfloat16,
                                                               device_map="auto")
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.model_name = model_name

    def run(self, premise, hypothesis):
        if self.openai:
            template = '''You will be given a document and a sentence. You will need to determine if the document provides enough evidence to support the sentence.
Document: {premise}


Sentence: {hypothesis}

Does the document provide enough evidence to support the sentence? 
Return Yes or No.
'''
            text_input = template.format(premise=premise, hypothesis=hypothesis)
            output = get_openai_completion(text_input, model_name=self.model_name)
            return 1 if "yes" in output.lower() else 0

        else:
            input_text = f"premise: {premise} hypothesis: {hypothesis}"
            input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True, padding=False,
                                       ).input_ids.to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(input_ids, max_new_tokens=10)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            result = 1 if result.startswith("1") else 0
            return result

