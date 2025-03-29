import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class JeopardyRAG:
    """
    Retrieval-Augmented Generation for Jeopardy clues.
    Uses a vector database to retrieve similar clues to enhance factual accuracy.
    """
    
    def __init__(
        self, 
        data_path: str = "/Users/sid/Projects/code/JeopardyLLM/data",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None
    ):
        self.data_path = Path(data_path)
        self.index_path = Path(index_path) if index_path else self.data_path / "vector_index"
        self.embedding_model_name = embedding_model_name
        
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.index = None
        self.clues_df = None
        
    def load_jeopardy_data(self):
        """Load all Jeopardy data from TSV files."""
        all_dfs = []
        
        tsv_files = list(self.data_path.glob("*.tsv"))
        logger.info(f"Found {len(tsv_files)} TSV files in {self.data_path}")
        
        for file_path in tsv_files:
            try:
                df = pd.read_csv(file_path, sep="\t")
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        if all_dfs:
            self.clues_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Total clues loaded: {len(self.clues_df)}")
            return self.clues_df
        else:
            logger.error("No data loaded")
            return None
    
    def build_vector_index(self, force_rebuild=False):
        """Build or load a FAISS vector index of Jeopardy clues."""
        index_file = self.index_path / "jeopardy_faiss_index.bin"
        metadata_file = self.index_path / "jeopardy_metadata.json"
        
        if not force_rebuild and index_file.exists() and metadata_file.exists():
            try:
                logger.info(f"Loading existing vector index from {index_file}")
                self.index = faiss.read_index(str(index_file))
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                if self.clues_df is None:
                    self.load_jeopardy_data()
                    
                logger.info(f"Loaded vector index with {self.index.ntotal} vectors")
                return self.index
            except Exception as e:
                logger.error(f"Error loading existing index: {str(e)}")
                logger.info("Will rebuild index")
        
        if self.clues_df is None:
            self.load_jeopardy_data()
            if self.clues_df is None:
                logger.error("No data available to build index")
                return None
        
        self.index_path.mkdir(exist_ok=True, parents=True)
        
        clues = self.clues_df['answer'].fillna("").tolist()
        logger.info(f"Encoding {len(clues)} clues with {self.embedding_model_name}")
        
        batch_size = 128
        embeddings = []
        
        for i in tqdm(range(0, len(clues), batch_size)):
            batch = clues[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Saving index with {self.index.ntotal} vectors to {index_file}")
        faiss.write_index(self.index, str(index_file))
        
        with open(metadata_file, 'w') as f:
            metadata = {
                "embedding_model": self.embedding_model_name,
                "num_vectors": self.index.ntotal,
                "dimension": dimension,
                "created_at": str(pd.Timestamp.now())
            }
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Vector index built successfully")
        return self.index
    
    def retrieve_similar_clues(self, query: str, k: int = 5) -> pd.DataFrame:
        """Retrieve the k most similar clues to the query."""
        if self.index is None:
            self.build_vector_index()
            
        if self.index is None:
            logger.error("No vector index available for retrieval")
            return pd.DataFrame()
            
        query_embedding = self.embedding_model.encode([query])
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        similar_clues = self.clues_df.iloc[indices[0]]
        
        similar_clues = similar_clues.copy()
        similar_clues['distance'] = distances[0]
        
        return similar_clues
    
    def generate_rag_prompt(self, query: str, k: int = 3) -> str:
        """Generate a RAG-enhanced prompt for the Jeopardy model."""
        similar_clues = self.retrieve_similar_clues(query, k=k)
        
        context = []
        for _, row in similar_clues.iterrows():
            context_item = {
                "category": row.get("category", ""),
                "clue": row.get("answer", ""),
                "response": row.get("question", "")
            }
            context.append(context_item)
        
        rag_prompt = f"""<bos><start_of_turn>user
# Instructions
# You are Alex Trebek hosting the current season of Jeopardy! You will provide a clue (the 'answer' in Jeopardy terms), and a contestant will respond with the correct 'question'.
# Use the context below to help with factual accuracy.

Context (similar clues from Jeopardy history):
{json.dumps(context, indent=2)}

Please present this clue: "{query}"

<end_of_turn> <start_of_turn>model"""
        
        return rag_prompt
