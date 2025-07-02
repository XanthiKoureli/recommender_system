from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
import json
from typing import List
from metapub import PubMedFetcher
import random
import matplotlib.pyplot as plt
import numpy as np
from backend.abstract_retrieval.pubmed_retriever import PubMedAbstractRetriever
from components.chat_prompts import chat_prompt_template, qa_template, get_type_instructions, format_documents_for_prompt
from components.llm import llm
from bert_score import score
from backend.data_repository.local_storage import LocalJSONStore
from backend.rag_pipeline.chromadb_rag import ChromaDbRag
from backend.rag_pipeline.embeddings import embeddings_function
from components.chat_utils import ChatAgent
from rouge_score import rouge_scorer
from pprint import pprint
from langchain_core.documents.base import Document
import re
from collections import defaultdict


#  Initialize
data_repository = LocalJSONStore(storage_folder_path="backend/data")
rag_client = ChromaDbRag(persist_directory="backend/chromadb_storage", embeddings=embeddings_function)
chat_agent = ChatAgent(prompt=chat_prompt_template, llm=llm)
chain = qa_template | llm

  
def evaluate_article_retrieval(predicted_pmids, gold_pmids, k=5):
    predicted_top_k = set(predicted_pmids[:k])
    gold_pmids_int = set(int(pmid) for pmid in gold_pmids)
    retrieved = predicted_top_k.intersection(gold_pmids_int)
    recall = len(retrieved) / len(gold_pmids_int) if gold_pmids_int else 0
    return recall


def plot_recall_distribution(recall_scores, save_path="recall_distribution.png"):
    if not recall_scores:
        print("⚠️  No recall scores to plot. Exiting.")
        return
    
    recall_array = np.array(recall_scores)

    mean_recall = np.mean(recall_array)
    median_recall = np.median(recall_array)
    std_recall = np.std(recall_array)

    print(f"😎 Mean Recall@5: {mean_recall:.4f}")
    print(f"😎 Median Recall@5: {median_recall:.4f}")
    print(f"😎 Std Dev Recall@5: {std_recall:.4f}")

    plt.figure(figsize=(8,5))
    plt.hist(recall_array, bins=10, color='skyblue', edgecolor='black')
    plt.title("Distribution of Recall@5 Scores")
    plt.xlabel("Recall@5")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)

    plt.savefig(save_path)  
    plt.close()  

    print(f"Histogram saved as {save_path}")


def evaluate_answer_with_bertscore(predicted, gold: list[str]):
    def normalize_text(text):
        import re
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    predicted_norm = normalize_text(predicted)
    bert_precisions = []
    bert_recalls = []
    bert_f1s = []
    rouge1_f1s = []
    rouge2_f1s = []
    rougeL_f1s = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for gold_answer in gold:
        gold_norm = normalize_text(gold_answer)
        P, R, F1 = score([predicted_norm], [gold_norm], lang='en', rescale_with_baseline=False)

        bert_precisions.append(max(0, P.item()))
        bert_recalls.append(max(0, R.item()))
        bert_f1s.append(max(0, F1.item()))

        rouge_scores = scorer.score(gold_norm, predicted_norm)
        rouge1_f1s.append(rouge_scores['rouge1'].fmeasure)
        rouge2_f1s.append(rouge_scores['rouge2'].fmeasure)
        rougeL_f1s.append(rouge_scores['rougeL'].fmeasure)

    return {
        '😎 bert_precision': max(bert_precisions),
        '😎 bert_recall': max(bert_recalls),
        '😎 bert_f1': max(bert_f1s),
        '😎 rouge1_f1': max(rouge1_f1s),
        '😎 rouge2_f1': max(rouge2_f1s),
        '😎 rougeL_f1': max(rougeL_f1s),
    }


def retrieve_answer(question, retrieved_abstracts, query_simplified, type_instruction):

    query_id = data_repository.save_dataset(
        retrieved_abstracts,
        question,
        query_simplified,
        None)
    
    documents = data_repository.create_document_list(retrieved_abstracts)
    
    rag_client.create_vector_index_for_user_query(documents, query_id)

    vector_index = rag_client.get_vector_index_by_user_query(query_id)
        
    retrieved_document_tuples = chat_agent.retrieve_documents(vector_index, question)
    retrieved_documents = [doc_tuple[0] for doc_tuple in retrieved_document_tuples]  
    
    formatted_abstracts = format_documents_for_prompt(retrieved_documents)

    type_instructions = get_type_instructions(type_instruction)

    llm_answer = chain.invoke({
        "question": question,
        "retrieved_abstracts": formatted_abstracts,
        "type_instructions": type_instructions
    }).content
    
    return llm_answer

def load_dataset(path):
    print(f"🔷 Loading dataset from {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset file not found: {path}")
        raise
    except json.JSONDecodeError:
        print("❌ ERROR: Failed to parse JSON.")
        
    
def main():
    print("🔷 Starting...")
    question_threshold = 10
    recall_scores =[]
    all_bertscores = defaultdict(list)
    pubmed_client = PubMedAbstractRetriever(PubMedFetcher())

    dataset = load_dataset("evaluation/training12b_new.json")
        
    all_questions = dataset['questions']
    if not all_questions:
        print("⚠️ No questions found in dataset.")
        return
    
    random.seed(11) 
    sampled_questions = random.sample(all_questions, question_threshold) 
    print(f"🔷 Sampled {len(sampled_questions)} question(s)")
    
    
    for idx, question in enumerate(sampled_questions): 
        print(f"🔶 Processing question #{idx + 1}...")
        body = question['body']
        documents = question['documents']
        ideal_answer = question['ideal_answer']
        type_instruction = question['type']
        
        gold_pmids = set(pmid.split("/")[-1] for pmid in documents) 
        print(f"🔸 Gold PMIDs: {gold_pmids}")
        
        retrieved_pmids = pubmed_client.get_abstract_data(
                                scientist_question=body, input_is_medical_notes=False)

        abstracts, query_simplified = retrieved_pmids

        if not abstracts:
            print("WARNING : abstracts are not found")
            continue 
        
        
        predicted_pmids = [abstract.pmid for abstract in abstracts]
        print(f"🔸 Predicted PMIDs: {predicted_pmids}")
        
        recall_scores.append(evaluate_article_retrieval(predicted_pmids, gold_pmids, k=5))
                
        predicted_answer = retrieve_answer(body, abstracts, query_simplified, type_instruction)
        
        print("🟩 Predicted answer:")
        print(predicted_answer)
        print("🟨 Ideal answer:")
        print(ideal_answer)
        
        score = evaluate_answer_with_bertscore(predicted_answer, ideal_answer)
        
        for k, v in score.items():
            all_bertscores[k].append(v)
            

        
    print("✅ Article retrival evaluation")
    plot_recall_distribution(recall_scores)
    
    print("📊 Average BERTScore:")
    avg_bertscores = {k: sum(v) / len(v) for k, v in all_bertscores.items()}
    pprint(avg_bertscores, indent=2)


    print("✅ Done.")
        


if __name__ == "__main__":
    main()
    
    
# How to run it.   python -m evaluation.evaluation

# Mean Recall@5: 0.1628
# Median Recall@5: 0.0000
# Std Dev Recall@5: 0.3430