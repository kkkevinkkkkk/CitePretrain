from utils import read_jsonl, save_jsonl
from evaluation import (RepliQAEvaluation,
                        ShortFormCitationEvaluator, LongformCitationEvaluator,
                        f1_score_token_level, SciQAGEvaluation,
                        ASQAEvaluation, Eli5Evaluation)
import fire
import numpy as np
from copy import deepcopy
import joblib
import pandas as pd
import re

pd.options.mode.copy_on_write = True
class Evaluator:
    def __init__(self, prediction_file="", df=None, recalibration_model_path=None):
        self.prediction_file = prediction_file
        if df is not None:
            self.df = df
        else:
            self.df = read_jsonl(self.prediction_file)
        self.model_name = self.prediction_file.split("/")[-1].split("_predictions")[0]
        self.retriever = None
        self.ner = None
        self.extractor = None
        self.multiple_choice = True if "multiple_choice" in self.prediction_file else False

        self.dataset_name = None
        self.dataset_evaluator = None
        if "selfeval" in self.prediction_file or "self_eval" in self.prediction_file:
            self.dataset_name = "selfeval"
        elif "repliqa" in self.prediction_file:
            self.dataset_name = "repliqa"
            self.dataset_evaluator = RepliQAEvaluation()
        elif "sciqag" in self.prediction_file:
            self.dataset_name = "sciqag"
            self.dataset_evaluator = SciQAGEvaluation()
        elif "asqa" in self.prediction_file:
            self.dataset_name = "asqa"
            self.dataset_evaluator = ASQAEvaluation()
        elif "eli5" in self.prediction_file:
            self.dataset_name = "eli5"
            self.dataset_evaluator = Eli5Evaluation()


        else:
            raise NotImplementedError()

        self.citation_evaluator = None
        rag_citation = False
        if "run_citation" in self.prediction_file:
            ctype = 0
            knowledge_source = "cite_pretrain"
            if "ctype" in self.prediction_file:
                pattern = r"ctype-(\d+)"
                ctype = int(re.findall(pattern, self.prediction_file)[0])
            if "run_citation_rag" in self.prediction_file:
                if "4.4.0" in self.prediction_file:
                    ctype = 1
                elif "4.4.1" in self.prediction_file:
                    ctype = 1
                    self.df["generated_text"] = self.df["generated_text"].apply(lambda x: x.split("## Final answer")[-1].strip())
                else:
                    ctype = 3
                    knowledge_source = ""
                    rag_citation = True




            if self.dataset_name in ["asqa", "eli5"]:
                self.citation_evaluator = LongformCitationEvaluator(self.df, ctype=ctype, knowledge_source_name=knowledge_source)
            elif self.dataset_name in [ "repliqa", "sciqag"]:
                self.citation_evaluator = ShortFormCitationEvaluator(self.df, ctype=ctype, knowledge_source_name=knowledge_source, rag_citation=rag_citation)
            else:
                raise NotImplementedError()


        self.recalibration_model_path = recalibration_model_path
        self.recalibration_model = joblib.load(self.recalibration_model_path) if self.recalibration_model_path is not None else None

    def get_model_answer(self, eval_item):
        generated_text = eval_item["generated_text"]

        if self.citation_evaluator:
            generated_text = self.citation_evaluator.get_model_answer(eval_item)

        model_answer = generated_text


        return model_answer






    def get_self_consistency(self, eval_item, norm_method=""):
        model_answer = self.get_model_answer(eval_item)
        consistency_scores = []
        normalize = True
        if len(eval_item["other_answers"]) == 0:
            return 1.0
        for other_answer in eval_item["other_answers"]:
            eval_item_ = deepcopy(eval_item)
            eval_item_["generated_text"] = other_answer
            if model_answer in ["A", "B", "C", "D"]:
                normalize = False
            other_answer_ = self.get_model_answer(eval_item_)
            consistency_scores.append(f1_score_token_level(model_answer, other_answer_, normalize=normalize))
        return np.mean(consistency_scores)


    def process_predictions(self, norm_method=""):

        self.df["model_answer"] = self.df.apply(self.get_model_answer, axis=1)
        self.df["model_other_answers"] = self.df.apply(
            lambda x: [self.get_model_answer({"generated_text": x}) for x in x["other_answers"]], axis=1)

        return self.df

    def calculate_ece_score(self, confidence_scores, correctness_scores, n_bins=10):
        # calculate ece score
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_boundaries[-1] += 1e-4
        ece_score = 0.0
        total_sample = len(confidence_scores)
        total_props = 0
        for bin_idx in range(n_bins):
            bin_mask = (confidence_scores >= bin_boundaries[bin_idx]) & (confidence_scores < bin_boundaries[bin_idx + 1])
            bin_confidence_scores = confidence_scores[bin_mask]
            bin_correctness_scores = correctness_scores[bin_mask]
            if len(bin_confidence_scores) == 0:
                continue
            bin_prop = len(bin_confidence_scores) / total_sample
            total_props += bin_prop
            bin_err = np.abs(np.mean(bin_correctness_scores) - np.mean(bin_confidence_scores))
            ece_score += bin_err * bin_prop
        return ece_score


    def evaluate(self, norm_method="", sc_norm_method=""):
        self.multiple_choice = True if "multiple_choice" in self.prediction_file else False

        self.df = self.process_predictions(norm_method=norm_method)

        if self.citation_evaluator:
            self.df = self.citation_evaluator.evaluate_citations()

        self.df = self.dataset_evaluator.evaluate_dataset(self.df)


        # calculate confidence score and ece score
        self.df["self_consistency"] = self.df.apply(self.get_self_consistency,
                                                    args=[sc_norm_method], axis=1)
        # recalibrate self_consistency if the recalibration model is provided
        if self.recalibration_model:
            self.df["self_consistency"] = self.recalibration_model.transform(self.df["self_consistency"])

        if "seq_log_prob_average" in self.df.columns:
            self.df["seq_prob"] = self.df.apply(lambda x: np.exp(x["seq_log_prob"]), axis=1)
            self.df["seq_prob_avg"] = self.df.apply(lambda x: np.exp(x["seq_log_prob_average"]), axis=1)
            self.df["seq_prob_filtered"] = self.df.apply(lambda x: np.exp(x["seq_log_prob_filtered"]), axis=1)

        self.df["correctness_score"] = self.df["score"]
        self.df["confidence_score"] = self.df["self_consistency"]

        ece_score = self.calculate_ece_score(self.df["confidence_score"].values, self.df["correctness_score"].values)
        self.df["ece_score"] = [ece_score] * len(self.df)

        scores_summary = self.df["scores_summary"][0]
        return scores_summary


def main(prediction_file, recalibration_model_path=None):
    evaluator = Evaluator(prediction_file, recalibration_model_path=recalibration_model_path)

    scores = evaluator.evaluate()
    print(scores)

    df = evaluator.df
    save_jsonl(df, prediction_file + ".score")



if __name__ == "__main__":
    fire.Fire(main)

