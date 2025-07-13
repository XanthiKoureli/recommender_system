from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class PubMedBERTEmbeddings:
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts):
        return self._get_embeddings(texts)

    def embed_query(self, text):
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token as embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        # Normalize embeddings
        normed = embeddings / embeddings.norm(dim=1, keepdim=True)
        return normed.numpy()
