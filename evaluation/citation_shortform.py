from .utils import metric_max_over_ground_truths, f1_score
import numpy as np
import re
class ShortFormCitationEvaluator:
    def __init__(self, df, ctype=0, knowledge_source_name="", rag_citation=False):
        self.df = df
        self.ctype = ctype
        self.SOT_TOKEN = "<|reserved_special_token_0|>"
        self.EOT_TOKEN = "<|reserved_special_token_1|>"
        self.knowledge_source_name = knowledge_source_name
        self.rag_citation = rag_citation

    def get_model_answer(self, eval_item):
        if self.ctype == 1:
            return eval_item["generated_text"].split(self.SOT_TOKEN)[0].strip()
        else:
            return eval_item["generated_text"].split("[1]")[0].strip()
        return eval_item["generated_text"]

    def get_generated_citations_rag(self, eval_item):
        generated_text = eval_item["generated_text"]
        sent = generated_text
        ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1
        num_docs = len(eval_item["docs"]) if "docs" in eval_item else 0
        ref = [r for r in ref if r < num_docs]  # Filter out references that are not in the docs
        ref = sorted(list(set(ref)))
        generated_citations = [eval_item["docs"][r]["title"] for r in ref]

        return generated_citations


    def get_generated_citations(self, eval_item):
        if self.rag_citation:
            return self.get_generated_citations_rag(eval_item)
        SOT_TOKEN_ESC = re.escape(self.SOT_TOKEN)
        EOT_TOKEN_ESC = re.escape(self.EOT_TOKEN)
        generated_text = eval_item["generated_text"]
        if self.ctype == 1:
            pattern = rf"{SOT_TOKEN_ESC}(.*?){EOT_TOKEN_ESC}"
            matches = re.findall(pattern, eval_item["generated_text"])
            generated_citations = [match.strip() for match in matches]

        else:
            generated_citations = eval_item["generated_text"].split("[1]")
            if len(generated_citations) == 1:
                generated_citations = generated_citations[0]
            else:
                generated_citations = "[1] " + eval_item["generated_text"].split("[1]")[1].strip()
            generated_citations = generated_citations.split("\n")
            # remove the [number] from the beginning of each citation
            generated_citations = [citation for citation in generated_citations if "]" in citation]
            generated_citations = [citation.split("]")[1].strip() for citation in generated_citations]
            # remove special tokens
            generated_citations = [citation.replace(self.SOT_TOKEN, "").replace(self.EOT_TOKEN, "") for citation in
                                   generated_citations]
        if len(generated_citations) == 0:
            generated_citations = [eval_item["generated_text"]]
        return generated_citations

    def evaluate_citations_item(self, eval_item):
        generated_citations = self.get_generated_citations(eval_item)
        citations = eval_item["citations"]
        generated_text = eval_item["generated_text"]
        correct_citations = 0
        f1s = []

        # if len(generated_citations) > 1:
        #     print("haha")
        for citation in generated_citations:
            f1 = metric_max_over_ground_truths(f1_score, citation, citations)
            f1s.append(f1)
            # if f1 > 0.5:
            if f1 > 0.9999:
                correct_citations += 1

        if len(generated_citations) == 0:
            f1s = [0.0]

        citation_precision = correct_citations / len(generated_citations) if len(generated_citations) > 0 else 0.0
        # citation_recall = correct_citations / len(citations)
        citation_recall = 1 if correct_citations > 0 else 0.0
        citation_f1 = 2 * citation_precision * citation_recall / (citation_precision + citation_recall) if citation_precision + citation_recall > 0 else 0.0
        precision_f1 = np.mean(f1s)
        citation_best_acc = np.max(f1s) >= 0.5
        eval_item["generated_citations"] = generated_citations
        eval_item["citation_precision"] = citation_precision
        eval_item["citation_recall"] = citation_recall
        eval_item["citation_f1"] = citation_f1
        eval_item["citation_precision_f1"] = precision_f1
        eval_item["citation_precision_f1_max"] = np.max(f1s)
        eval_item["citation_best_acc"] = citation_best_acc
        return eval_item

    def evaluate_citations(self):
        if "citations" not in self.df.columns:
            self.df["citations"] = self.df["title"].apply(lambda x: [x])
        self.df = self.df.apply(self.evaluate_citations_item, axis=1)
        citation_precision = self.df["citation_precision"].mean()
        citation_recall = self.df["citation_recall"].mean()
        citation_f1 = self.df["citation_f1"].mean()
        citation_precision_f1 = self.df["citation_precision_f1"].mean()
        citation_precision_f1_max = self.df["citation_precision_f1_max"].mean()
        citation_best_acc = self.df["citation_best_acc"].mean()
        print(f"citation precision: {citation_precision}, citation recall: {citation_recall}, citation f1: {citation_f1}, \
                    citation_precision_f1: {citation_precision_f1}, citation_precision_f1_max: {citation_precision_f1_max}, citation_best_acc: {citation_best_acc}")
        return self.df
