
### 2️⃣ Keyword Coverage
Measures overlap between query keywords and retrieved context.

### Refusal Behavior
If:
- Confidence < threshold OR
- Keyword coverage < threshold

The system refuses generation and suggests rephrasing.

This prevents unsupported answers when the document lacks relevant information.

---

## 📸 Demo

### High-Confidence QA
![QA Example](screenshots/qa_example.png)

---

### Refusal Case (Low Coverage)
![Refusal Example](screenshots/refusal_example.png)

---

### Quiz Generation Mode
![Quiz Example](screenshots/quiz_example.png)

---

## 🛠 Tech Stack

- Python
- LangChain
- FAISS (Vector Store)
- OpenAI Embeddings
- OpenAI Chat Model
- Google Colab

---

## ⚙️ Features

- Multi-mode agent routing (QA / Summary / Quiz)
- Strict citation enforcement (page-level)
- Context-only answering
- Evidence preview from top retrieved chunks
- Confidence and coverage diagnostics

---

## 📌 Example Queries

- "What JUnit 5 dependencies do I add and where?"
- "Summarize what Lab 6 requires."
- "Make a quiz on CityListTest requirements."
- "What neural network architecture should I use?" → triggers refusal

---

## 🔐 Security

API keys are never stored in the notebook.
Keys are provided at runtime using `getpass()`.

---

## 📈 Why This Matters

Large Language Models frequently hallucinate when context is insufficient.

This project demonstrates:

- Retrieval-grounded generation
- Guard-based refusal logic
- Practical mitigation of hallucination risks
- Production-oriented LLM system design

---

## 📂 Project Structure


