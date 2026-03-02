from .utils import normalize_answer

import numpy as np
import pandas as pd
from transformers import pipeline

import logging
import collections


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


QA_MODEL="gaotianyu1350/roberta-large-squad"


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False

def compute_str_em(row):
    loc_acc = []
    for qa_pair in row['qa_pairs']:
        loc_acc.append(exact_presence(qa_pair['short_answers'], row["model_answer"]))
    acc = np.mean(loc_acc)
    hit = int(np.mean(loc_acc) == 1)
    return {"Str-EM": acc, "Str-Hit": hit}

def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

class ASQAEvaluation:
    def __init__(self):
        self.qa_pipeline = None
        self.score_key = "qa_f1"


    def compute_qa(self, row):
        """Compute QA-based accuracy.
        Args:
            data: requires filed `qa_pairs/short_answers` and `output`
        Returns:
            QA metrics (QA-EM, QA-F1, QA-Hit)
        """

        if 'qa_pairs' not in row or row['qa_pairs'] is None:
            logger.warning("No QA pairs found in data")
            return {
                'QA-EM': 0,
                'QA-F1': 0,
                'QA-Hit': 0,
            }


        if self.qa_pipeline is None:
            # Load model
            # logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
            self.qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=0)
            # logger.info("Done")

        # Get prediction
        # logger.info("Computing the QA-based accuracy...")
        item = row

        question = [qa_pair['question'] for qa_pair in item['qa_pairs']]
        context = item['model_answer'] if len(item['model_answer']) > 0 else " "
        results = self.qa_pipeline(question=question, context=context, handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0

        for idx, res in enumerate(results):
            answers = item["qa_pairs"][idx]["short_answers"]
            prediction = res["answer"]

            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            loc_counter += 1

        qa_em = loc_em / loc_counter
        qa_f1 = loc_f1 / loc_counter
        qa_hit = loc_em == loc_counter

        return {"QA-EM": qa_em, "QA-F1": qa_f1, "QA-Hit": qa_hit}


    def evaluate_single_answer(self, row):
        prediction = row['model_answer']
        ground_truths = [row["answer"]]
        str_scores = compute_str_em(row)
        qa_scores = self.compute_qa(row)
        scores = {
            "str_em": str_scores["Str-EM"],
            "qa_em": qa_scores["QA-EM"],
            "qa_f1": qa_scores["QA-F1"],
            "qa_hit": qa_scores["QA-Hit"],
            "str_hit": str_scores["Str-Hit"]
        }
        return scores




    def evaluate_row(self, row):
        model_answer0 = row['model_answer']
        other_answers = row["other_answers"]
        model_answers = [model_answer0] + other_answers
        scores = []
        for model_answer in model_answers:
            model_answer_ = model_answer
            row["model_answer"] = model_answer_
            scores.append(self.evaluate_single_answer(row)[self.score_key])


        row["model_answer"] = model_answer0
        row["expected_correctness"] = np.mean(scores)
        return row


    def evaluate_dataset(self, df):
        results = df.apply(self.evaluate_single_answer, axis=1)
        results = pd.DataFrame(results.tolist())

        df["score"] = results.apply(lambda x: x[self.score_key], axis=1)
        scores = [x.to_dict() for i, x in results[["qa_em", "qa_f1", "qa_hit", "str_em", "str_hit"]].iterrows()]

        qa_em = results.apply(lambda x: x["qa_em"], axis=1).mean()
        qa_f1 = results.apply(lambda x: x["qa_f1"], axis=1).mean()
        qa_hit = results.apply(lambda x: x["qa_hit"], axis=1).mean()
        str_em = results.apply(lambda x: x["str_em"], axis=1).mean()
        str_hit = results.apply(lambda x: x["str_hit"], axis=1).mean()
        total_scores = {"qa_f1": qa_f1, "str_em": str_em}

        df["scores"] = scores
        df["scores_summary"] = [total_scores] * len(df)

        return df




