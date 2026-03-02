import numpy as np
import pandas as pd

import logging
from utils import NLIModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class Eli5Evaluation:
    def __init__(self):
        self.score_key = "claim_score"
        self.nli_model = NLIModel()


    def evaluate_single_answer(self, row):
        prediction = row['model_answer']
        claims = row["claims"]
        entail = 0
        for claim in claims:
            entail += self.nli_model.run(prediction, claim)
        claim_score = entail / len(claims)
        scores = {
            "claim_score": claim_score
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
        scores = [x.to_dict() for i, x in results[["claim_score"]].iterrows()]

        claim_score = results.apply(lambda x: x["claim_score"], axis=1).mean()


        total_scores = {"claim_score": claim_score}

        df["scores"] = scores
        df["scores_summary"] = [total_scores] * len(df)

        return df




