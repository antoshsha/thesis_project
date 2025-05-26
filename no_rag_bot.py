import os
import json
import random

import ragas.metrics
from dotenv import load_dotenv

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from langchain.chat_models import ChatOpenAI
from ragas.metrics import (
    ResponseRelevancy,
    AnswerAccuracy,
    FactualCorrectness,
    SemanticSimilarity,
    NonLLMStringSimilarity,
    BleuScore,
    RougeScore,
)


def load_qa(path="questions.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_questions(all_qa, num, seed):
    rng = random.Random(seed)
    if num is None or num >= len(all_qa):
        return all_qa
    return rng.sample(all_qa, num)


def prompt_int(prompt, default=None):
    s = input(prompt).strip()
    if s == "":
        return default
    try:
        return int(s)
    except ValueError:
        print("Please enter a valid integer or blank.")
        return prompt_int(prompt, default)


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4.1-mini",
        temperature=0.0,
    )

    SYSTEM_PREFIX = (
        "You are an expert in Ukrainian governmental services. "
        "Answer *only* about those services. Be extremely concise and to the point.\n\n"
    )

    all_qa = load_qa("questions.json")
    num = prompt_int("How many questions to test? (blank=all): ", default=None)
    seed = prompt_int("Random seed? (blank=42): ", default=42)
    qa_list = sample_questions(all_qa, num, seed)

    samples = []
    for i, qa in enumerate(qa_list, start=1):
        prompt = SYSTEM_PREFIX + qa["question"]
        answer = llm.predict(prompt).strip()
        samples.append(
            SingleTurnSample(
                user_input=qa["question"],
                retrieved_contexts=[],
                response=answer,
                reference=qa["answer"],
            )
        )
        print(f"[{i}/{len(qa_list)}] Q: {qa['question']}")
        print(f"→ A: {answer}")
        print(f"✓ ref: {qa['answer']}\n")

    dataset = EvaluationDataset(samples=samples)

    print("Running Ragas evaluation\n")
    metrics = [
        ragas.metrics.answer_relevancy,
        ragas.metrics.answer_similarity,
        ragas.metrics.answer_correctness,
        ResponseRelevancy(),
        FactualCorrectness(),
        SemanticSimilarity(),
        NonLLMStringSimilarity(),
        BleuScore(),
        RougeScore(),
        AnswerAccuracy(),
    ]
    results = evaluate(metrics=metrics, dataset=dataset, llm=llm)
    print(results)


if __name__ == "__main__":
    main()
