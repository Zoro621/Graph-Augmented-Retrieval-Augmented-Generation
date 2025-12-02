"""
Dataset Loading Utilities for Graph-Augmented RAG
Supports FEVER, HotpotQA, TruthfulQA, and custom datasets
"""

import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import requests
from pathlib import Path
import pandas as pd


@dataclass
class QueryDataset:
    """Container for query datasets"""
    queries: List[str]
    ground_truths: List[str]
    contexts: List[List[str]]  # Supporting documents for each query
    metadata: List[Dict[str, Any]]


class DatasetLoader:
    """
    Unified loader for various QA datasets
    """
    
    def __init__(self, cache_dir: str = "./data"):
        """
        Initialize dataset loader
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_fever(
        self,
        split: str = "train",
        max_samples: int = 100
    ) -> QueryDataset:
        """
        Load FEVER dataset from:
            data/fever/train.jsonl
            data/fever/dev.jsonl
            data/fever/test.jsonl
        """

        print(f"Loading FEVER {split} split...")

        # NEW: support folder-based structure
        folder_path = self.cache_dir / "fever"
        file_path = folder_path / f"{split}.jsonl"

        # Fallback: support old “fever_dev.jsonl” format
        fallback_path = self.cache_dir / f"fever_{split}.jsonl"

        # Choose correct file
        if file_path.exists():
            chosen_path = file_path
        elif fallback_path.exists():
            chosen_path = fallback_path
        else:
            raise FileNotFoundError(
                f"❌ FEVER file not found.\n"
                f"Expected one of:\n"
                f"  - {file_path}\n"
                f"  - {fallback_path}\n"
                f"Place your dataset inside: data/fever/{split}.jsonl"
            )

        queries = []
        ground_truths = []
        contexts = []
        metadata = []

        with open(chosen_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break

                data = json.loads(line)

                claim = data.get("claim", "")
                label = data.get("label", "NOT ENOUGH INFO")
                evidence = data.get("evidence", [])

                # Extract evidence strings
                context_docs = []
                for evidence_set in evidence:
                    for item in evidence_set:
                        if len(item) >= 3:
                            context_docs.append(item[2])

                queries.append(claim)
                ground_truths.append(label)
                contexts.append(context_docs)
                metadata.append({
                    "id": data.get("id", i),
                    "verifiable": data.get("verifiable", ""),
                    "label": label
                })

        print(f"✓ Loaded {len(queries)} FEVER samples.")

        return QueryDataset(
            queries=queries,
            ground_truths=ground_truths,
            contexts=contexts,
            metadata=metadata
        )

        
    # def load_fever(
    #     self,
    #     split: str = "dev",
    #     max_samples: int = 100
    # ) -> QueryDataset:
    #     """
    #     Load FEVER (Fact Extraction and VERification) dataset
        
    #     Args:
    #         split: 'train' or 'dev'
    #         max_samples: Maximum number of samples to load
            
    #     Returns:
    #         QueryDataset object
    #     """
    #     print(f"Loading FEVER {split} split...")
        
    #     file_path = self.cache_dir / f"fever_{split}.jsonl"
        
    #     if not file_path.exists():
    #         print(f"FEVER dataset not found at {file_path}")
    #         print("Please download from: https://fever.ai/resources.html")
    #         print(f"Save as: {file_path}")
    #         return None
        
    #     queries = []
    #     ground_truths = []
    #     contexts = []
    #     metadata = []
        
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         for i, line in enumerate(f):
    #             if i >= max_samples:
    #                 break
                
    #             data = json.loads(line)
                
    #             # Extract claim as query
    #             claim = data.get('claim', '')
    #             label = data.get('label', 'NOT ENOUGH INFO')
    #             evidence = data.get('evidence', [])
                
    #             # Build context from evidence
    #             context_docs = []
    #             for evidence_set in evidence:
    #                 for evidence_item in evidence_set:
    #                     if len(evidence_item) >= 3:
    #                         # Evidence format: [annotation_id, evidence_id, wikipedia_url, sent_id]
    #                         context_docs.append(evidence_item[2] if len(evidence_item) > 2 else "")
                
    #             queries.append(claim)
    #             ground_truths.append(label)
    #             contexts.append(context_docs)
    #             metadata.append({
    #                 "id": data.get('id', i),
    #                 "verifiable": data.get('verifiable', ''),
    #                 "label": label
    #             })
        
    #     print(f"✓ Loaded {len(queries)} samples from FEVER")
        
    #     return QueryDataset(
    #         queries=queries,
    #         ground_truths=ground_truths,
    #         contexts=contexts,
    #         metadata=metadata
    #     )
    
    def load_hotpotqa(
        self,
        split: str = "dev",
        max_samples: int = 100,
        distractor: bool = True
    ) -> QueryDataset:
        """
        Load HotpotQA dataset
        
        Args:
            split: 'train' or 'dev'
            max_samples: Maximum number of samples
            distractor: Whether to use distractor setting
            
        Returns:
            QueryDataset object
        """
        print(f"Loading HotpotQA {split} split...")
        
        filename = f"hotpot_{split}_{'distractor' if distractor else 'fullwiki'}_v1.json"
        file_path = self.cache_dir / filename
        
        if not file_path.exists():
            print(f"HotpotQA dataset not found at {file_path}")
            print("Please download from: https://hotpotqa.github.io/")
            print(f"Save as: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = []
        ground_truths = []
        contexts = []
        metadata = []
        
        for i, item in enumerate(data[:max_samples]):
            query = item.get('question', '')
            answer = item.get('answer', '')
            
            # Extract supporting facts as context
            context_docs = []
            for doc_title, doc_sents in item.get('context', []):
                context_docs.extend(doc_sents)
            
            queries.append(query)
            ground_truths.append(answer)
            contexts.append(context_docs)
            metadata.append({
                "id": item.get('_id', i),
                "type": item.get('type', ''),
                "level": item.get('level', '')
            })
        
        print(f"✓ Loaded {len(queries)} samples from HotpotQA")
        
        return QueryDataset(
            queries=queries,
            ground_truths=ground_truths,
            contexts=contexts,
            metadata=metadata
        )
    
    def load_truthfulqa(
        self,
        max_samples: int = 100
    ) -> QueryDataset:
        """
        Load TruthfulQA dataset
        
        Args:
            max_samples: Maximum number of samples
            
        Returns:
            QueryDataset object
        """
        print("Loading TruthfulQA...")
        
        file_path = self.cache_dir / "truthfulqa.csv"
        
        if not file_path.exists():
            print(f"TruthfulQA dataset not found at {file_path}")
            print("Please download from: https://github.com/sylinrl/TruthfulQA")
            print(f"Save as: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        df = df.head(max_samples)
        
        queries = df['Question'].tolist()
        ground_truths = df['Best Answer'].tolist()
        
        # TruthfulQA doesn't have explicit contexts
        contexts = [[] for _ in range(len(queries))]
        
        metadata = [
            {
                "category": row.get('Category', ''),
                "source": row.get('Source', '')
            }
            for _, row in df.iterrows()
        ]
        
        print(f"✓ Loaded {len(queries)} samples from TruthfulQA")
        
        return QueryDataset(
            queries=queries,
            ground_truths=ground_truths,
            contexts=contexts,
            metadata=metadata
        )
    
    def load_custom_json(
        self,
        file_path: str,
        query_key: str = "question",
        answer_key: str = "answer",
        context_key: str = "context"
    ) -> QueryDataset:
        """
        Load custom JSON dataset
        
        Expected format:
        [
            {
                "question": "...",
                "answer": "...",
                "context": ["doc1", "doc2", ...]
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            query_key: Key for query/question
            answer_key: Key for ground truth answer
            context_key: Key for context documents
            
        Returns:
            QueryDataset object
        """
        print(f"Loading custom dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = []
        ground_truths = []
        contexts = []
        metadata = []
        
        for i, item in enumerate(data):
            queries.append(item.get(query_key, ''))
            ground_truths.append(item.get(answer_key, ''))
            
            context = item.get(context_key, [])
            if isinstance(context, str):
                context = [context]
            contexts.append(context)
            
            # Store all other fields as metadata
            meta = {k: v for k, v in item.items() 
                   if k not in [query_key, answer_key, context_key]}
            metadata.append(meta)
        
        print(f"✓ Loaded {len(queries)} samples from custom dataset")
        
        return QueryDataset(
            queries=queries,
            ground_truths=ground_truths,
            contexts=contexts,
            metadata=metadata
        )
    
    def create_sample_dataset(self) -> QueryDataset:
        """
        Create a sample dataset for testing
        
        Returns:
            QueryDataset with Einstein facts
        """
        print("Creating sample dataset...")
        
        queries = [
            "When was Albert Einstein born?",
            "What is Einstein's most famous equation?",
            "Why did Einstein receive the Nobel Prize?",
            "When did Einstein move to the United States?",
            "What does general relativity describe?",
            "Where did Einstein die?",
            "What year was special relativity published?",
            "What position was Einstein offered in 1952?"
        ]
        
        ground_truths = [
            "Albert Einstein was born on March 14, 1879.",
            "Einstein's most famous equation is E=mc².",
            "Einstein received the Nobel Prize for his explanation of the photoelectric effect.",
            "Einstein moved to the United States in 1933.",
            "General relativity describes gravity as a curvature of spacetime.",
            "Einstein died in Princeton, New Jersey.",
            "Special relativity was published in 1905.",
            "Einstein was offered the presidency of Israel in 1952."
        ]
        
        # Shared context documents
        base_contexts = [
            "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
            "Einstein's theory of special relativity introduced the equation E=mc².",
            "In 1921, Einstein received the Nobel Prize in Physics for explaining the photoelectric effect.",
            "Einstein emigrated to the United States in 1933 and worked at Princeton.",
            "General relativity describes gravity as spacetime curvature caused by mass.",
            "Einstein died on April 18, 1955, in Princeton, New Jersey.",
            "Special relativity was published in 1905.",
            "Einstein was offered the presidency of Israel in 1952 but declined."
        ]
        
        # Each query gets all contexts (simulating retrieval)
        contexts = [base_contexts for _ in queries]
        
        metadata = [{"source": "sample", "id": i} for i in range(len(queries))]
        
        print(f"✓ Created sample dataset with {len(queries)} queries")
        
        return QueryDataset(
            queries=queries,
            ground_truths=ground_truths,
            contexts=contexts,
            metadata=metadata
        )
    
    def save_dataset(
        self,
        dataset: QueryDataset,
        output_path: str
    ):
        """
        Save dataset to JSON format
        
        Args:
            dataset: QueryDataset to save
            output_path: Path to save file
        """
        data = []
        
        for q, gt, ctx, meta in zip(
            dataset.queries,
            dataset.ground_truths,
            dataset.contexts,
            dataset.metadata
        ):
            data.append({
                "query": q,
                "ground_truth": gt,
                "contexts": ctx,
                "metadata": meta
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Dataset saved to {output_path}")


class DatasetDownloader:
    """
    Helper to download datasets
    """
    
    DATASET_URLS = {
        "fever_train": "https://fever.ai/download/fever/train.jsonl",
        "fever_dev": "https://fever.ai/download/fever/dev.jsonl",
        "hotpotqa_train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "hotpotqa_dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    }
    
    @staticmethod
    def download_dataset(
        dataset_name: str,
        output_path: str,
        chunk_size: int = 8192
    ):
        """
        Download a dataset
        
        Args:
            dataset_name: Name of dataset (e.g., 'fever_dev')
            output_path: Where to save
            chunk_size: Download chunk size
        """
        if dataset_name not in DatasetDownloader.DATASET_URLS:
            print(f"Unknown dataset: {dataset_name}")
            print(f"Available: {list(DatasetDownloader.DATASET_URLS.keys())}")
            return False
        
        url = DatasetDownloader.DATASET_URLS[dataset_name]
        
        print(f"Downloading {dataset_name} from {url}...")
        print(f"Saving to {output_path}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='')
            
            print(f"\n✓ Download complete: {output_path}")
            return True
            
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = DatasetLoader(cache_dir="./data")
    
    # Create sample dataset for testing
    print("\n" + "="*60)
    print("DATASET LOADER - DEMONSTRATION")
    print("="*60)
    
    sample_data = loader.create_sample_dataset()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Queries: {len(sample_data.queries)}")
    print(f"   Ground truths: {len(sample_data.ground_truths)}")
    print(f"   Context docs per query: {len(sample_data.contexts[0])}")
    
    print(f"\n📝 Sample Query:")
    print(f"   Query: {sample_data.queries[0]}")
    print(f"   Ground Truth: {sample_data.ground_truths[0]}")
    print(f"   Context: {sample_data.contexts[0][0][:100]}...")
    
    # Save sample dataset
    loader.save_dataset(sample_data, "sample_dataset.json")
    
    print("\n" + "="*60)
    print("To download real datasets:")
    print("="*60)
    print("1. FEVER: https://fever.ai/resources.html")
    print("2. HotpotQA: https://hotpotqa.github.io/")
    print("3. TruthfulQA: https://github.com/sylinrl/TruthfulQA")
    print("\nOr use DatasetDownloader.download_dataset() helper")