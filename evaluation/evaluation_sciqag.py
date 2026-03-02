from .utils import recall_score, metric_max_over_ground_truths

import numpy as np
from utils import multi_process_map
import pandas as pd

from pipeline import CustomLLM
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class ScoreDict(BaseModel):
    score: int = Field(..., description="The score for the criterion")
    comment: str = Field(..., description="The reason for the score")
class Answer(BaseModel):
    accuracy: ScoreDict = Field(..., description="The accuracy of the answer in relation to the question")
    completeness: ScoreDict = Field(..., description="The completeness of the answer")
    reasonableness: ScoreDict = Field(..., description="The reasonableness of the answer")



USE_GPT4 = True
EVAL_MODEL_NAME = "gpt-4.1"
_pipe = None

def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = CustomLLM(model_name=EVAL_MODEL_NAME)
    return _pipe

eval_prompt = '''For this task, you are provided with a question- answer pair. Evaluate the quality of answer on the following three criteria and record your evaluations in a score ranging from 1 to 5 for each criterion and provide reasons for assigning the score:
1. **Accuracy**: Score the accuracy of the answer in relation to the question. A score of 5 means the answer is fully accurate. This involves checking the accuracy of any claims or statements made in the text, and verifying that they are supported by evidence. While a score of 1 indicates significant inaccuracies.
2. **Completeness**: Rate how comprehensive the answer is. A score of 5 indicates that the answer addresses all key points of the question and includes sufficient background and supporting details and evidence. A score of 1 means the answer is largely incomplete.
3. **Reasonableness**: Evaluate the logical consistency and reasonableness of the answer. A score of 5 indicates that the answer is logically sound with no contradictions; a score of 1 indi- cates that the answer contains major contradictions.
Provide the scores in a dictionary output. The dictionary is with three keys (name of 3 criterions). The value is a tuple (score and comment).
**Example JSON Output:**
{format_instruction}

**Input Sections:**
- **Question & Answer Pair**: {qa}
**Output:**'''

class SciQAGEvaluation:

    @staticmethod
    def evaluate_single_answer(row):
        prediction = row['model_answer']
        ground_truths = [row["answer"]]
        search_result = re.search(r"([^.!?]*[.!?])\s*$", prediction)
        if search_result is not None:
            prediction = search_result.group(1)
        else:
            prediction = prediction


        recall = metric_max_over_ground_truths(
            recall_score, prediction, ground_truths)


        question = row["question"]
        gpt_score = False
        parser = PydanticOutputParser(pydantic_object=Answer)
        format_instruction = parser.get_format_instructions()
        acc = 0
        completeness = 0
        reasonableness = 0
        if USE_GPT4:
            qa = '\nQuestion: ' + question + '\nAnswer: ' + prediction
            text_input = eval_prompt.format(qa=qa, format_instruction=format_instruction)
            acc, completeness, reasonableness = SciQAGEvaluation.get_gpt_score(text_input, parser)


        scores = {"recall": recall, "accuracy": acc, "completeness": completeness, "reasonableness": reasonableness}

        return scores
    @staticmethod
    def get_gpt_score(text_input, parser, max_retry=3):
        for i in range(max_retry):
            try:
                output = _get_pipe()(text_input)[0]["generated_text"]
                answer_scores = parser.parse(output)
                return answer_scores.accuracy.score / 5.0, answer_scores.completeness.score / 5.0, answer_scores.reasonableness.score / 5.0
            except Exception:
                continue
        return 0, 0, 0

    @staticmethod
    def evaluate_row(row):
        model_answer0 = row['model_answer']
        other_answers = row["other_answers"]
        model_answers = [model_answer0] + other_answers
        scores = []
        for model_answer in model_answers:
            model_answer_ = model_answer
            row["model_answer"] = model_answer_
            score_key = "recall" if not USE_GPT4 else "accuracy"
            scores.append(SciQAGEvaluation.evaluate_single_answer(row)[score_key])

        row["model_answer"] = model_answer0
        row["expected_correctness"] = np.mean(scores)
        return row


    @staticmethod
    def evaluate_dataset(df):

        results = multi_process_map(df, SciQAGEvaluation.evaluate_single_answer, num_proc=64)

        df["score"] = results.apply(lambda x: x["recall"] if not USE_GPT4 else x["accuracy"], axis=1)
        scores = [x.to_dict() for i, x in results[["recall",  "accuracy", "completeness", "reasonableness"]].iterrows()]
        recall = results.apply(lambda x: x["recall"], axis=1).mean()
        accuracy = results.apply(lambda x: x["accuracy"], axis=1).mean()
        completeness = results.apply(lambda x: x["completeness"], axis=1).mean()
        reasonableness = results.apply(lambda x: x["reasonableness"], axis=1).mean()
        total_scores = {"recall": recall, "accuracy": accuracy, "completeness": completeness, "reasonableness": reasonableness}
        df["scores"] = scores
        df["scores_summary"] = [total_scores] * len(df)
        return df




