import os
from dotenv import load_dotenv
from functions import load_questions, sample, prompt_int
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from langchain.chat_models import ChatOpenAI
import ragas.metrics as m


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

    all_qa = load_questions("questions.json")
    num = prompt_int("How many questions to test? (blank=all): ", default=None)
    seed = prompt_int("Random seed? (blank=42): ", default=42)
    qa_list = sample(all_qa, num, seed)

    samples = []
    for i, qa in enumerate(qa_list, start=1):
        prompt = SYSTEM_PREFIX + qa["question"]
        answer = llm.predict(prompt).strip()
        print(f"[{i}/{len(qa_list)}] Q: {qa['question']}")
        print(f"→ A: {answer}")
        print(f"✓ ref: {qa['answer']}\n")

        samples.append(
            SingleTurnSample(
                user_input=qa["question"],
                retrieved_contexts=[],
                response=answer,
                reference=qa["answer"],
            )
        )

    dataset = EvaluationDataset(samples=samples)

    print("Running Ragas evaluation\n")
    metrics = [
        m.answer_relevancy,
        m.answer_similarity,
        m.answer_correctness,
        m.ResponseRelevancy(),
        m.FactualCorrectness(),
        m.SemanticSimilarity(),
        m.NonLLMStringSimilarity(),
        m.BleuScore(),
        m.RougeScore(),
        m.AnswerAccuracy(),
    ]

    results = evaluate(metrics=metrics, dataset=dataset, llm=llm)
    print(results)


if __name__ == "__main__":
    main()
