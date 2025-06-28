import json
from backend.abstract_retrieval.pubmed_retriever import PubMedAbstractRetriever
from metapub import PubMedFetcher
import random
import matplotlib.pyplot as plt
import numpy as np

def evaluate_retrieval(predicted_pmids, gold_pmids, k=5):
    predicted_top_k = set(predicted_pmids[:k])
    gold_pmids_int = set(int(pmid) for pmid in gold_pmids)
    retrieved = predicted_top_k.intersection(gold_pmids_int)
    recall = len(retrieved) / len(gold_pmids_int) if gold_pmids_int else 0
    return recall

def plot_recall_distribution(recall_scores, save_path="recall_distribution.png"):
    recall_array = np.array(recall_scores)

    mean_recall = np.mean(recall_array)
    median_recall = np.median(recall_array)
    std_recall = np.std(recall_array)

    print(f"Mean Recall@5: {mean_recall:.4f}")
    print(f"Median Recall@5: {median_recall:.4f}")
    print(f"Std Dev Recall@5: {std_recall:.4f}")

    plt.figure(figsize=(8,5))
    plt.hist(recall_array, bins=10, color='skyblue', edgecolor='black')
    plt.title("Distribution of Recall@5 Scores")
    plt.xlabel("Recall@5")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)

    plt.savefig(save_path)  
    plt.close()  

    print(f"Histogram saved as {save_path}")
    
    
def main():
    pubmed_client = PubMedAbstractRetriever(PubMedFetcher())

    with open("evaluation/training12b_new.json") as f:
        dataset = json.load(f)
        

    all_questions = dataset['questions']
    
    random.seed(42) 
    sampled_questions = random.sample(all_questions, 5) 
    recall_scores =[]
    
    for question in sampled_questions: 
 
        body = question['body']
        documents = question['documents']

        gold_pmids = set(pmid.split("/")[-1] for pmid in documents) 

        retrieved_pmids = pubmed_client.get_abstract_data(
                                scientist_question=body, input_is_medical_notes=False)

        abstracts, query = retrieved_pmids

        predicted_pmids = [abstract.pmid for abstract in abstracts]
        
        recall = evaluate_retrieval(predicted_pmids, gold_pmids,k=5)
        recall_scores.append(recall)
        
        
    plot_recall_distribution(recall_scores)
        


if __name__ == "__main__":
    main()
    
    
# How to run it.   python -m evaluation.evaluation

# Mean Recall@5: 0.1628
# Median Recall@5: 0.0000
# Std Dev Recall@5: 0.3430