from qdrant_client import QdrantClient, models
from datasets import load_dataset
import pandas as pd
import re
from transformers import AutoTokenizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, SparseVector
from nltk.corpus import stopwords
import torch

class HybridSearchIndexer:
    def __init__(self, collection_name="hybrid_search", url="http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
        self.dense_model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr', device=self.device)
        self.tfidf = TfidfVectorizer(max_features=10000, stop_words=list(stopwords.words('turkish')))
        
    def setup_collection(self):
        """Create or recreate the Qdrant collection"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": models.VectorParams(size=768, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))}
        )
        print(f"✓ Collection '{self.collection_name}' created")

    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('turkish'))
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        
        return ' '.join(words).strip()

    def chunk_text(self, text, max_tokens=256, overlap_tokens=50):
        """Smart text chunking with word boundary preservation"""
        if not text or not text.strip():
            return []
        
        words = text.split()
        if not words:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, word in enumerate(words):
            word_tokens = len(self.tokenizer.encode(word, add_special_tokens=False))
            
            if current_tokens + word_tokens <= max_tokens:
                current_chunk.append(word)
                current_tokens += word_tokens
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Create overlap
                overlap_start = max(0, len(current_chunk) - overlap_tokens)
                current_chunk = current_chunk[overlap_start:] + [word]
                current_tokens = sum(len(self.tokenizer.encode(w, add_special_tokens=False)) for w in current_chunk)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]

    def process_documents(self, texts, ids, titles):
        """Process documents into chunks with metadata"""
        all_chunks, metadata = [], []
        
        for i, text in enumerate(texts):
            cleaned = self.clean_text(text)
            if not cleaned:
                continue
                
            chunks = self.chunk_text(cleaned)
            if not chunks:
                chunks = [cleaned[:500]]
            
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    'original_doc_id': ids[i],
                    'original_title': titles[i],
                    'chunk_index': idx,
                    'total_chunks': len(chunks)
                })
        
        return all_chunks, metadata

    def create_vectors(self, chunks):
        """Create both dense and sparse vectors"""
        print(f"Creating vectors on {self.device}...")
        
        # Dense vectors
        dense_vectors = self.dense_model.encode(
            chunks, 
            show_progress_bar=True, 
            batch_size=64 if self.device == "cuda" else 32
        )
        
        # Sparse vectors (TF-IDF)
        sparse_matrix = self.tfidf.fit_transform(chunks)
        
        # Save TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf, f)
        
        return dense_vectors, sparse_matrix

    def to_sparse_vector(self, sparse_row):
        """Convert scipy sparse row to Qdrant sparse vector format"""
        return SparseVector(
            indices=sparse_row.indices.tolist(),
            values=sparse_row.data.tolist()
        )

    def upload_to_qdrant(self, chunks, metadata, dense_vectors, sparse_matrix, batch_size=100):
        """Upload chunks to Qdrant in batches"""
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            
            points = []
            for i in range(start_idx, end_idx):
                points.append(PointStruct(
                    id=i,
                    vector={
                        "dense": dense_vectors[i].tolist(),
                        "sparse": self.to_sparse_vector(sparse_matrix[i])
                    },
                    payload={
                        "chunk_text": chunks[i],
                        **metadata[i]
                    }
                ))
            
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Batch {batch_idx + 1}/{total_batches} uploaded")

    def index_dataset(self, dataset_name="umarigan/turkish_wikipedia", sample_size=50):
        """Main indexing pipeline"""
        print("Loading dataset...")
        ds = load_dataset(dataset_name)
        data = ds["train"].select(range(sample_size))
        
        # Setup collection
        self.setup_collection()
        
        # Process documents
        print("Processing documents...")
        chunks, metadata = self.process_documents(
            data["text"], data["id"], data["title"]
        )
        print(f"Created {len(chunks)} chunks from {sample_size} documents")
        
        # Create vectors
        dense_vectors, sparse_matrix = self.create_vectors(chunks)
        
        # Upload to Qdrant
        print("Uploading to Qdrant...")
        self.upload_to_qdrant(chunks, metadata, dense_vectors, sparse_matrix)
        
        print("✓ Indexing completed successfully!")
        return len(chunks)

# Usage
if __name__ == "__main__":
    indexer = HybridSearchIndexer()
    total_chunks = indexer.index_dataset(sample_size=50)
    print(f"\n=== SUMMARY ===")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Device used: {indexer.device}")
    print(f"Collection: {indexer.collection_name}")