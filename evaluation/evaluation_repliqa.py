from .utils import (metric_max_over_ground_truths,
                    exact_match_score_relax, recall_score, precision_score)

import numpy as np
from .evaluation_freshqa import evaluate_with_llm
from utils import multi_process_map
import re
import pandas as pd

from pipeline import CustomLLM

USE_GPT4 = True
EVAL_MODEL_NAME = "gpt-4.1"
_pipe = None
save_budget = False

def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = CustomLLM(model_name=EVAL_MODEL_NAME)
    return _pipe

class RepliQAEvaluation:

    @staticmethod
    def evaluate_single_answer(row):
        prediction = row['model_answer']
        ground_truths = [row["answer"]]
        search_result = re.search(r"([^.!?]*[.!?])\s*$", prediction)
        if search_result is not None:
            prediction = search_result.group(1)
        else:
            prediction = prediction

        em_for_this_question_relax = metric_max_over_ground_truths(
            exact_match_score_relax, prediction, ground_truths)

        recall = metric_max_over_ground_truths(
            recall_score, prediction, ground_truths)
        precision = metric_max_over_ground_truths(
            precision_score, prediction, ground_truths)

        question = row["question"]
        gpt_score = False
        if USE_GPT4:
            if save_budget and precision < 0.15 and recall < 0.15:
                gpt_score = False
            else:
                gpt_score = evaluate_with_llm(_get_pipe(), question, ground_truths, prediction)

        scores = {"recall": float(recall), "gpt_score": float(gpt_score), "precision": float(precision), "em_relax": float(em_for_this_question_relax)}

        return scores

    @staticmethod
    def evaluate_row(row):
        model_answer0 = row['model_answer']
        other_answers = row["other_answers"]
        model_answers = [model_answer0] + other_answers
        scores = []
        for model_answer in model_answers:
            model_answer_ = model_answer
            row["model_answer"] = model_answer_
            score_key = "recall" if not USE_GPT4 else "gpt_score"
            scores.append(RepliQAEvaluation.evaluate_single_answer(row)[score_key])

        row["model_answer"] = model_answer0
        row["expected_correctness"] = np.mean(scores)
        return row


    @staticmethod
    def evaluate_dataset(df):

        results = multi_process_map(df, RepliQAEvaluation.evaluate_single_answer, num_proc=64)

        df["score"] = results.apply(lambda x: x["recall"] if not USE_GPT4 else x["gpt_score"], axis=1)
        scores = [x.to_dict() for i, x in results[["recall", "gpt_score", "precision"]].iterrows()]
        # em_relax = results.apply(lambda x: x["em_relax"], axis=1).mean()
        recall = results.apply(lambda x: x["recall"], axis=1).mean()
        gpt_score = results.apply(lambda x: x["gpt_score"], axis=1).mean()
        precision = results.apply(lambda x: x["precision"], axis=1).mean()
        total_scores = {"recall": recall, "gpt_score": gpt_score, "precision": precision}
        df["scores"] = scores
        df["scores_summary"] = [total_scores] * len(df)
        return df




