# 🔍 Biomedical QA Evaluation Pipeline

This project evaluates a Biomedical Question Answering (QA) system using BERTScore and ROUGE metrics. 
It retrieves relevant PubMed abstracts, generates answers via LLM, and assesses performance based on document recall and answer similarity.


## 🚀 How It Works

1. **Load dataset** from `training12b_new.json`.
2. **Retrieve abstracts** for each question using the PubMed API.
3. **Evaluate document retrieval** via Recall@5 against gold PMIDs.
4. **Generate an answer** with a large language model (LLM).
5. **Score answer quality** using BERTScore and ROUGE.
6. **Visualize recall** distribution using matplotlib.

---

## 📊 Metrics Used

- **BERTScore**: Semantic similarity using contextual embeddings.
- **ROUGE**: Lexical overlap via ROUGE-1, ROUGE-2, and ROUGE-L.
- **Recall@5**: Measures how many relevant documents were retrieved in the top 5.

---


## 🧪 Run the Evaluation

```bash
python -m evaluation.evaluation
```

This will output:

- Predicted vs ideal answers
- BERT & ROUGE scores
- Document Recall@5 distribution
- A histogram saved as `recall_distribution.png`

---

## 📁 Input Format

Sample format for `training12b_new.json`:

```json
{
  "questions": [
    {
      "body": "What antigens are associated with blood type?",
      "documents": ["https://pubmed.ncbi.nlm.nih.gov/12345678", "..."],
      "ideal_answer": [
        "ABO antigens are highly abundant in many human cell types...",
        "The blood group antigens associated with blood type are..."
      ],
      "type": "factoid"
    }
  ]
}
```

---

## 📈 Output Example

```
- Article retrival evaluation

😎 Mean Recall@5: 0.1628
😎 Median Recall@5: 0.0000
😎 Std Dev Recall@5: 0.3430

- Score answer quality

📊 Average BERTScore:
{ 
	'😎 bert_f1': 0.8731821378072103,
	'😎 bert_precision': 0.8414338628451029,
	'😎 bert_recall': 0.9099662105242411,
	'😎 rouge1_f1': 0.32691244261887337,
	'😎 rouge2_f1': 0.22444409008422497,
	'😎 rougeL_f1': 0.2057003214067522
}
```

---

## ✅ Dependencies

All requirements are listed in `requirements.txt` (generated via `pip freeze`).

---




## 🧠 1. BERTScore (BERT-based Semantic Similarity)

**What it does:**
- Uses a pretrained language model (**BERT**) to compare the **meaning** of words in your answer vs the reference.
- It looks at **token embeddings**, not just surface-level word matches.
- It’s **semantic** — if you rephrase the answer with different words but the same meaning, BERTScore will still be high.

**Your BERTScore results:**
- 😎 **bert_precision** = `0.841`: How much of your answer matches meaningfully with the gold answer.
- 😎 **bert_recall** = `0.910`: How much of the gold answer is covered by your answer.
- 😎 **bert_f1** = `0.873`: A balance of both precision and recall.

✅ **Interpretation:**  
Your model is doing **very well** semantically. The generated answer captures the **meaning** of the gold answer almost completely. Excellent!

---

## 📝 2. ROUGE (Surface-Level Overlap)

**What it does:**
- ROUGE = *Recall-Oriented Understudy for Gisting Evaluation*
- Measures **n-gram overlap** — i.e., how many **words or sequences** of words in your output match the gold answer exactly.
- It’s more **strict** — it cares about **exact wording and order**, not just meaning.

**Your ROUGE results:**
- 😎 **rouge1_f1** = `0.327`: Unigram (single word) overlap.
- 😎 **rouge2_f1** = `0.224`: Bigram (2-word sequence) overlap.
- 😎 **rougeL_f1** = `0.206`: Longest common subsequence overlap.

🚫 **Interpretation:**  
These scores are **moderate to low**, meaning:
- You’re **not using the exact same words or phrasing** as the ideal answer.
- But that’s **okay** — because your **BERTScore is high**, so **meaning is preserved**.


---


# Embedding Models Explanation

The code selects different **embedding models** based on the `MODEL_NAME` setting, each designed to generate vector representations of text for downstream tasks like semantic search or question answering.

- **OpenAIEmbeddings**:  
  Uses OpenAI’s API (e.g., GPT models) to produce general-purpose embeddings. These are versatile and strong for a wide variety of NLP tasks.

- **MistralAIEmbeddings**:  
  Uses embeddings from Mistral AI’s models, which are designed for efficient and high-quality text understanding.

- **PubMedBERTEmbeddings**:  
  Based on BERT models pre-trained specifically on biomedical literature from PubMed. It excels at understanding medical and scientific text.

- **SapBERTEmbeddings**:  
  Built on a specialized BERT model called SapBERT, which is fine-tuned for biomedical entity representations, making it particularly effective for tasks requiring precise medical terminology understanding.



>I chose **SapBERT** because, during testing, it consistently produced better results on the specific domain or dataset I was working with — likely due to its biomedical focus and superior ability to capture semantic relationships in that context.
>
>
> The selected embeddings are then passed to the `ChromaDbRag` client, which uses them to index and retrieve information effectively from a vector database.
