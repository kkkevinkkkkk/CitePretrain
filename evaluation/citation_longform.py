
from .utils import metric_max_over_ground_truths, f1_score
import numpy as np
import re
from utils import NLIModel, Retriever
from nltk import sent_tokenize
from copy import deepcopy
from tqdm import tqdm
from .knowledge_source import KnowledgeSource
import pandas as pd
from utils import multi_process_map
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from pipeline import get_custom_llm
tqdm.pandas()

EXTRACTION_MODEL = "gpt-4.1"


class FactWithSources(BaseModel):
    fact: str = Field(..., description="The self-contained fact claim")
    sources: list[str] = Field(..., description="The sources of the fact claim")
class FactsWithSources(BaseModel):
    facts: list[FactWithSources] = Field(..., description="The self-contained fact claims and their sources")

template_extract_facts = '''
You will be given a question and an answer. In the answer, each fact claim is followed by one citation or more citations. The citation is a document title marked by <|reserved_special_token_0|> and <|reserved_special_token_1|>. You need to extract the sentences their corresponding citations. Your need to rephrase each sentence so that they are self-contained. 

Question: {question}
Answer: {answer}

Your output should follow the format below:
{format_instruction}
'''

USE_GPT_TO_EXTRACT = True
class LongformCitationEvaluator:
    def __init__(self, df, nli_model_name=None, ctype=0, knowledge_source_name=""):
        self.df = df
        nli_model_name = "gpt-4o-mini"
        # nli_model_name = "gpt-4o"
        nli_model_name = "google/t5_11b_trueteacher_and_anli"
        self.openai = True if "gpt" in nli_model_name else False
        self.nli_model_name = nli_model_name
        self.nli_model = NLIModel(nli_model_name) if nli_model_name else None
        self.retriever = Retriever() if nli_model_name and not self.openai else None
        # self.nli_model2 = NLIModel(nli_model_name2)
        self.ctype = ctype
        self.knowledge_source_name = knowledge_source_name
        self.max_len_per_passage = 4096 if self.openai else 1024
        self.ks = None if knowledge_source_name == "" else KnowledgeSource()


    def get_model_answer(self, eval_item):
        SOT_TOKEN = "<|reserved_special_token_0|>"
        EOT_TOKEN = "<|reserved_special_token_1|>"
        if self.ctype == 0:
            SOT_TOKEN = "<source>"
            EOT_TOKEN = "</source>"
        SOT_TOKEN_ESC = re.escape(SOT_TOKEN)
        EOT_TOKEN_ESC = re.escape(EOT_TOKEN)

        pattern = rf"{SOT_TOKEN_ESC}(.*?){EOT_TOKEN_ESC}"
        if self.ctype == 2:
            pattern = r"\[\d+\]\s*<[^>]+>"

        if self.ctype == 3:
            pattern = r"\[\d+\]"

        generated_text = eval_item["generated_text"]
        cleaned_text = re.sub(pattern, "", eval_item["generated_text"], flags=re.DOTALL)

        return cleaned_text

    def remove_citations(self, sent):
        if self.ctype == 3:
            return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")
        else:
            raise NotImplementedError

    def get_generated_citations_rag(self, eval_item):
        generated_text = eval_item["generated_text"]
        sentences = sent_tokenize(generated_text)
        target_sentences = [self.remove_citations(sent) for sent in sentences]
        generated_citations = []
        for sent_id, sent in enumerate(sentences):
            target_sentence = target_sentences[sent_id]
            # Find references
            ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1
            num_docs = len(eval_item["docs"]) if "docs" in eval_item else 0
            ref = [r for r in ref if r < num_docs]  # Filter out references that are not in the docs
            titles = [eval_item["docs"][r]["title"] for r in ref]
            local_doc_ids = [r for r in ref]
            generated_citations.append({"sentence": target_sentence, "source": titles, "local_doc_ids": local_doc_ids})

        return generated_citations

    def search_docs(self, generated_citations):
        docs = []
        titles = []
        cur_wrong_local_doc_id = -1
        cur_local_doc_id = 0
        for i, citation in enumerate(generated_citations):
            # title = citation["source"]
            generated_citations[i]["local_doc_ids"] = []
            sources = citation["source"] if "source" in citation else []
            for title in sources:
                if title in titles:
                    idx = titles.index(title)
                    generated_citations[i]["local_doc_ids"] = [idx]
                    continue

                page = self.ks.get_page_by_title(title)
                if page is None:
                    generated_citations[i]["local_doc_ids"].append(cur_wrong_local_doc_id)
                    cur_wrong_local_doc_id -= 1
                else:
                    # text = "\n".join([para for para in page["text"]])
                    text = page['text']
                    docs.append({"title": title, "text": text})

                    titles.append(title)
                    generated_citations[i]["local_doc_ids"].append(cur_local_doc_id)
                    cur_local_doc_id += 1
        return docs, generated_citations




    def get_generated_citations(self, eval_item):
        # if use RAG then use the RAG citation extraction
        if self.nli_model and self.ks is None:
            return self.get_generated_citations_rag(eval_item)
        SOT_TOKEN = "<|reserved_special_token_0|>"
        EOT_TOKEN = "<|reserved_special_token_1|>"
        if self.ctype == 0:
            SOT_TOKEN = "<source>"
            EOT_TOKEN = "</source>"
        SOT_TOKEN_ESC = re.escape(SOT_TOKEN)
        EOT_TOKEN_ESC = re.escape(EOT_TOKEN)

        # pattern = rf"([A-Z].*?[.!?\"])\s*{SOT_TOKEN_ESC}(.*?){EOT_TOKEN_ESC}"
        # if self.ctype == 2:
        #     pattern = r"(.*?)\[\d+\]\s*<(.*?)>[.!?\"']"
        # text = eval_item["generated_text"]
        # matches = re.findall(pattern, text)
        # # Extract the supported sentences and sources
        # results = [(match[0].strip(), match[1].strip()) for match in matches]
        # generated_citations = [{"sentence": result[0], "source": result[1]} for result in results]

        sentence_src_block_pat = rf"""
            ([A-Z].*?[.!?"])                # group 1: the sentence
            (?:\s*{SOT_TOKEN_ESC}.*?{EOT_TOKEN_ESC})+   # one‑or‑more sources
        """
        single_source_pat = rf"{SOT_TOKEN_ESC}(.*?){EOT_TOKEN_ESC}"

        sent_rx = re.compile(sentence_src_block_pat, re.VERBOSE | re.S)
        src_rx = re.compile(single_source_pat, re.S)
        grouped_citations = []  # one dict per sentence with list of sources

        for m in sent_rx.finditer(eval_item["generated_text"]):
            sentence = m.group(1).strip()
            sources = [s.strip() for s in src_rx.findall(m.group(0))]
            grouped_citations.append({"sentence": sentence, "sources": sources})

        generated_citations = grouped_citations

        if USE_GPT_TO_EXTRACT:
            parser = PydanticOutputParser(pydantic_object=FactsWithSources)
            format_instruction = parser.get_format_instructions()

            text_input = template_extract_facts.format(question=eval_item['question'], answer=eval_item["generated_text"],
                                                       format_instruction=format_instruction)
            output = get_custom_llm(EXTRACTION_MODEL)(text_input)[0]['generated_text']
            try:
                facts = parser.parse(output)
                generated_citations = [{"sentence": fact.fact, "source": fact.sources} for fact in facts.facts]
            except Exception as e:
                print(f"Error in parsing the output: {output}, use the original extraction method: {e}")

        if self.ks is not None:
            docs, generated_citations = self.search_docs(generated_citations)
            eval_item["docs"] = docs
        return generated_citations

    def evaluate_citations_item(self, eval_item):
        generated_citations = self.get_generated_citations(eval_item)
        eval_item["generated_citations"] = generated_citations

        if self.nli_model is not None:
            eval_item = self.evaluate_citations_nli(eval_item)
        # generated_citations = eval_item["generated_citations"]

        answer_sources_key = "answer_sources" if "answer_sources" in eval_item else "cot_answer_sources"
        if answer_sources_key not in eval_item:
            return eval_item

        citations = eval_item[answer_sources_key]
        citations = [c for c in citations if c["source"] != ""]

        if len(generated_citations) == 0:
            generated_citations_only = []
        elif isinstance(generated_citations[0]["source"], list):
            generated_citations_only = []
            for c in generated_citations:
                if len(c["source"]) > 0:
                    generated_citations_only.extend(c["source"])
        else:
            generated_citations_only = [c["source"] for c in generated_citations]
        generated_citations_only = list(set(generated_citations_only))

        citations_only = [c["source"] for c in citations]
        additional_citations = eval_item.get("citations", [])
        citations_only.extend(additional_citations)
        citations_only = list(set(citations_only))

        if len(citations_only) == 0:
            avg_f1_precision = 0
        else:
            f1s = []
            for i, citation in enumerate(generated_citations_only):
                f1 = metric_max_over_ground_truths(f1_score, citation, citations_only)
                f1 = 1 if f1 > 0.5 else 0
                f1s.append(f1)
                # generated_citations[i]["f1"] = f1
            avg_f1_precision = np.mean(f1s)

        if len(generated_citations_only) == 0:
            avg_f1_recall = 0
        else:
            recall_f1s = []
            for citation in citations_only:
                f1 = metric_max_over_ground_truths(f1_score, citation, generated_citations_only)
                f1 = 1 if f1 > 0.5 else 0
                recall_f1s.append(f1)
            avg_f1_recall = np.mean(recall_f1s)


        eval_item["citation_precision_approx"] = avg_f1_precision
        eval_item["citation_recall_approx"] = avg_f1_recall
        return eval_item

    def run_nli(self, passages, sentence, top_k=5):
        max_len = self.max_len_per_passage
        if len(passages) == 0:
            return 0
        elif len(passages) == 1:
            passage = passages[0]
            words = passage.split()
            entails = []
            # if the passage is too long, split it into smaller parts and run NLI on each part
            partial_passages = [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]
            # to save computation, only run nli on the top k passages
            if len(partial_passages) > top_k and self.retriever is not None:
                partial_passages = self.retriever.retrieve_paragraph(partial_passages, sentence, k=top_k)
            for partial_passage in partial_passages:
                entail = self.nli_model.run(partial_passage, sentence)
                # entail2 = self.nli_model2.run(partial_passage, sentence)
                # if entail != entail2:
                #     print(f"{sentence},  entail: {entail}, entail2: {entail2}")
                entails.append(entail)
            return any(entails)
        else:
            # TODO handle multiple passages
            max_len_per_passage = max_len // len(passages)
            partial_passages = [" ".join(passage.split()[:max_len_per_passage]) for passage in passages]
            joint_passage = "\n".join(partial_passages)
            entail = self.nli_model.run(joint_passage, sentence)
            # entail2 = self.nli_model2.run(joint_passage, sentence)
            # if entail != entail2:
            #     print(f"entail: {entail}, entail2: {entail2}")
            return entail


    def evaluate_citations_nli(self, eval_item):
        assert self.nli_model is not None
        generated_text = eval_item["generated_text"]
        generated_citations = eval_item["generated_citations"]

        def _format_document(doc):
            """Format document for AutoAIS."""

            if "sent" in doc:
                # QA-extracted docs
                return "Title: %s\n%s" % (doc['title'], doc['sent'])
            else:
                return "Title: %s\n%s" % (doc['title'], doc['text'])


        multi_cite_sentence_num = 0
        multi_cite_sentence_support_num = 0
        single_cite_sentence_num = 0
        zero_cite_sentence_num = 0
        citation_num = 0
        correct_citation_num = 0

        multi_cite_sentence_overcite_num = 0
        supported_sentence_num = 0
        sentences = [citation["sentence"] for citation in generated_citations]

        for i, citation in enumerate(generated_citations):
            # title = citation["source"]
            sentence = citation["sentence"]


            local_doc_ids = citation["local_doc_ids"]
            local_doc_ids  = list(set(local_doc_ids))
            local_docs = eval_item["docs"]

            ref_num = len(local_doc_ids)
            citation_num += ref_num
            if ref_num > 1:
                multi_cite_sentence_num += 1
            elif ref_num == 1:
                single_cite_sentence_num += 1
            else:
                zero_cite_sentence_num += 1

            # joint_passage = "\n".join([_format_document(local_docs[doc_id]) for doc_id in local_doc_ids])
            joint_passages = [_format_document(local_docs[doc_id]) if doc_id >= 0 else "" for doc_id in local_doc_ids]
            joint_entail = self.run_nli(joint_passages, sentence)

            supported_sentence_num += joint_entail
            eval_item["generated_citations"][i]["supported"] = joint_entail
            if joint_entail and ref_num > 1:
                multi_cite_sentence_support_num += 1
                for doc_id in local_doc_ids:
                    passage = _format_document(local_docs[doc_id]) if doc_id >= 0 else ""
                    nli_result = self.run_nli([passage], sentence)
                    # nli_result = self.nli_model.run(passage, sentence)

                    # if single citation does not support the sentence
                    if not nli_result:
                        local_doc_ids_subset = deepcopy(local_doc_ids)
                        local_doc_ids_subset.remove(doc_id)
                        joint_passages = [_format_document(local_docs[doc_id]) if doc_id >= 0 else "" for doc_id in local_doc_ids_subset]
                        nli_result = self.run_nli(joint_passages, sentence)
                        # if without this citation, the sentence is still supported, then it is overcite
                        if nli_result:
                            multi_cite_sentence_overcite_num += 1
                        # if without this citation, the sentence is not supported, then it is correct
                        else:
                            correct_citation_num += 1
                    else:
                        correct_citation_num += 1
            else:
                correct_citation_num += joint_entail

        cited_sentence_num = multi_cite_sentence_num + single_cite_sentence_num
        citations_stats = {
            "sentence_num": len(sentences),
            "citation_num": citation_num,
            "supported_sentence_num": supported_sentence_num,
            "correct_citation_num": correct_citation_num,
            "multi_cite_sentence_num": multi_cite_sentence_num,
            "multi_cite_sentence_support_num": multi_cite_sentence_support_num,
            "multi_cite_sentence_overcite_num": multi_cite_sentence_overcite_num,
            "single_cite_sentence_num": single_cite_sentence_num,
            "zero_cite_sentence_num": zero_cite_sentence_num,
            "citation_precision": correct_citation_num / citation_num if citation_num > 0 else 0,
            "citation_recall": supported_sentence_num / len(sentences) if len(sentences) > 0 else 0,
            "support_rate_at_cited": supported_sentence_num / cited_sentence_num if cited_sentence_num > 0 else 0,
        }
        eval_item["citation_recall"] = citations_stats["citation_recall"]
        eval_item["citation_precision"] = citations_stats["citation_precision"]
        eval_item["support_rate_at_cited"] = citations_stats["support_rate_at_cited"]
        eval_item["citation_stats"] = citations_stats

        return eval_item

    def evaluate_citations(self):
        debug = False
        if debug:
            # self.df = self.df.iloc[496:]
            self.df = self.df[:10]
        # if self.openai:
        #     self.df = multi_process_map(self.df, self.evaluate_citations_item)
        # else:
        # self.df["generated_citations"] = multi_process_map(self.df, self.get_generated_citations)
        # self.df["generated_citations"] = self.df.progress_apply(self.get_generated_citations, axis=1)
        self.df = self.df.progress_apply(self.evaluate_citations_item, axis=1)
        if "citation_precision_approx" in self.df:
            citation_precision = self.df["citation_precision_approx"].mean()
            citation_recall = self.df["citation_recall_approx"].mean()
            print(f"approx citation precision: {citation_precision}, approx citation recall: {citation_recall}")


        macro_citation_precision = self.df["citation_precision"].mean()
        micro_citation_precision = self.df["citation_stats"].apply(lambda x: x["correct_citation_num"]).sum() / self.df["citation_stats"].apply(lambda x: x["citation_num"]).sum()
        macro_citation_recall = self.df["citation_recall"].mean()
        micro_citation_recall = self.df["citation_stats"].apply(lambda x: x["supported_sentence_num"]).sum() / self.df["citation_stats"].apply(lambda x: x["sentence_num"]).sum()
        macro_support_rate_at_cited = self.df["support_rate_at_cited"].mean()
        print(f"macro citation precision: {macro_citation_precision}, macro citation recall: {macro_citation_recall}, macro support rate at cited: {macro_support_rate_at_cited}")
        print(f"citation precision: {micro_citation_precision}, citation recall: {micro_citation_recall}")
        if debug:
            raise ValueError("debug")
        return self.df
