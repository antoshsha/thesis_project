
# thesis_project

A collection of experiments evaluating various Retrieval-Augmented Generation (RAG) techniques on a dataset of Ukrainian governmental services. Each script implements a different RAG variant (or baseline) and measures performance using [Ragas](https://github.com/openai/ragas) metrics.


## ⚙️ Prerequisites

- **Python 3.8+**
- **Environment variables**  
  Create a `.env` file in the project root:
  ini
  OPENAI_API_KEY=<your_openai_api_key>
  COHERE_API_KEY=<your_cohere_api_key>


* **Install dependencies**
  If there’s no `requirements.txt`, install manually:

  ```bash
  pip install \
    python-dotenv \
    openai \
    cohere \
    langchain \
    ragas \
    faiss-cpu \
    networkx \
    scikit-learn
  ```

---

## 🚀 Usage

1. **Clone & enter**

   ```bash
   git clone https://github.com/antoshsha/thesis_project.git
   cd thesis_project
   ```

2. **Run an experiment script**
   Each script prompts for:

   * **Number of questions** (or leave blank for all)
   * **Random seed** (default: 42)

   Example with the main FAISS + Cohere-reranker pipeline:

   ```bash
   python main.py
   ```

   Or try the graph‐based RAG:

   ```bash
   python Graph_RAG.py
   ```

3. **View results**
   After generating answers, each script runs a Ragas evaluation and prints a summary of metrics:

   * Answer Relevancy, Similarity, Correctness
   * BLEU, ROUGE, Semantic Similarity, etc.

---

## 📊 Customization

* **Adjust chunking**
  In scripts that load `dataset.json`, you can modify:

  ```python
  chunk_size = 500
  chunk_overlap = 50
  ```
* **Index directory**
  By default, FAISS indexes are stored under `faiss_index/`. Change `INDEX_DIR` in the scripts if needed.
* **Models & parameters**
  Edit `model_name`, `temperature`, `k` (retrieval size), and RAG-specific hyperparameters directly in each script.

---

## 🤝 Contributing

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Add some feature"`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

```

Feel free to adjust any paths, dependencies, or experiment parameters to suit your setup.
```
