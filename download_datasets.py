"""
Dataset Download and Setup Script
Automatically downloads FEVER, HotpotQA, and TruthfulQA datasets
"""

import os
import sys
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import zipfile


class DatasetDownloader:
    """
    Automated dataset downloader for research project
    """
    
    def __init__(self, base_dir: str = "./data"):
        """
        Initialize downloader
        
        Args:
            base_dir: Base directory for datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs
        self.urls = {
            "fever_train": "https://fever.ai/download/fever/train.jsonl",
            "fever_dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
            "fever_test": "https://fever.ai/download/fever/shared_task_test.jsonl",
            "hotpotqa_train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
            "hotpotqa_dev_distractor": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "hotpotqa_dev_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
        }
        
        print(f"✓ Dataset downloader initialized")
        print(f"  Base directory: {self.base_dir.absolute()}")
    
    def download_file(
        self,
        url: str,
        output_path: Path,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a file with progress bar
        
        Args:
            url: URL to download from
            output_path: Where to save the file
            chunk_size: Download chunk size in bytes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\nDownloading from: {url}")
            print(f"Saving to: {output_path}")
            
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Stream download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(output_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=output_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✓ Download complete: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def download_fever(self, splits: list = ["dev"]) -> bool:
        """
        Download FEVER dataset
        
        Args:
            splits: Which splits to download ('train', 'dev', 'test')
            
        Returns:
            True if all downloads successful
        """
        print("\n" + "="*60)
        print("DOWNLOADING FEVER DATASET")
        print("="*60)
        
        fever_dir = self.base_dir / "fever"
        fever_dir.mkdir(exist_ok=True)
        
        success = True
        for split in splits:
            url_key = f"fever_{split}"
            if url_key not in self.urls:
                print(f"⚠ Unknown split: {split}")
                continue
            
            output_file = fever_dir / f"{split}.jsonl"
            
            # Skip if already exists
            if output_file.exists():
                print(f"✓ {split}.jsonl already exists, skipping")
                continue
            
            success = success and self.download_file(
                self.urls[url_key],
                output_file
            )
        
        if success:
            print(f"\n✓ FEVER dataset ready in {fever_dir}")
        
        return success
    
    def download_hotpotqa(self, splits: list = ["dev_distractor"]) -> bool:
        """
        Download HotpotQA dataset
        
        Args:
            splits: Which splits to download 
                   ('train', 'dev_distractor', 'dev_fullwiki')
            
        Returns:
            True if all downloads successful
        """
        print("\n" + "="*60)
        print("DOWNLOADING HOTPOTQA DATASET")
        print("="*60)
        
        hotpot_dir = self.base_dir / "hotpotqa"
        hotpot_dir.mkdir(exist_ok=True)
        
        success = True
        for split in splits:
            url_key = f"hotpotqa_{split}"
            if url_key not in self.urls:
                print(f"⚠ Unknown split: {split}")
                continue
            
            output_file = hotpot_dir / f"{split}.json"
            
            # Skip if already exists
            if output_file.exists():
                print(f"✓ {split}.json already exists, skipping")
                continue
            
            success = success and self.download_file(
                self.urls[url_key],
                output_file
            )
        
        if success:
            print(f"\n✓ HotpotQA dataset ready in {hotpot_dir}")
        
        return success
    
    def download_truthfulqa(self) -> bool:
        """
        Download TruthfulQA dataset
        
        Note: TruthfulQA is hosted on GitHub
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("DOWNLOADING TRUTHFULQA DATASET")
        print("="*60)
        
        truthful_dir = self.base_dir / "truthfulqa"
        truthful_dir.mkdir(exist_ok=True)
        
        # Main TruthfulQA file
        url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
        output_file = truthful_dir / "TruthfulQA.csv"
        
        if output_file.exists():
            print(f"✓ TruthfulQA.csv already exists, skipping")
            return True
        
        success = self.download_file(url, output_file)
        
        if success:
            print(f"\n✓ TruthfulQA dataset ready in {truthful_dir}")
        
        return success
    
    def download_all(self) -> bool:
        """
        Download all datasets
        
        Returns:
            True if all successful
        """
        print("\n" + "="*60)
        print("DOWNLOADING ALL DATASETS")
        print("="*60)
        
        # Download FEVER (dev only to save space)
        fever_success = self.download_fever(splits=["dev"])
        
        # Download HotpotQA (dev distractor for multi-hop QA)
        hotpot_success = self.download_hotpotqa(splits=["dev_distractor"])
        
        # Download TruthfulQA
        truthful_success = self.download_truthfulqa()
        
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"FEVER: {'✓' if fever_success else '✗'}")
        print(f"HotpotQA: {'✓' if hotpot_success else '✗'}")
        print(f"TruthfulQA: {'✓' if truthful_success else '✗'}")
        
        return fever_success and hotpot_success and truthful_success
    
    def verify_datasets(self) -> dict:
        """
        Verify that datasets were downloaded correctly
        
        Returns:
            Dict with verification results
        """
        print("\n" + "="*60)
        print("VERIFYING DATASETS")
        print("="*60)
        
        results = {}
        
        # Check FEVER
        fever_dev = self.base_dir / "fever" / "dev.jsonl"
        results["fever"] = {
            "exists": fever_dev.exists(),
            "size": fever_dev.stat().st_size if fever_dev.exists() else 0,
            "path": str(fever_dev)
        }
        
        # Check HotpotQA
        hotpot_dev = self.base_dir / "hotpotqa" / "dev_distractor.json"
        results["hotpotqa"] = {
            "exists": hotpot_dev.exists(),
            "size": hotpot_dev.stat().st_size if hotpot_dev.exists() else 0,
            "path": str(hotpot_dev)
        }
        
        # Check TruthfulQA
        truthful = self.base_dir / "truthfulqa" / "TruthfulQA.csv"
        results["truthfulqa"] = {
            "exists": truthful.exists(),
            "size": truthful.stat().st_size if truthful.exists() else 0,
            "path": str(truthful)
        }
        
        # Print results
        for dataset, info in results.items():
            status = "✓" if info["exists"] else "✗"
            size_mb = info["size"] / (1024 * 1024)
            print(f"{status} {dataset.upper()}: {size_mb:.1f} MB")
            if info["exists"]:
                print(f"   Location: {info['path']}")
        
        return results
    
    def get_dataset_info(self) -> dict:
        """
        Get information about available datasets
        
        Returns:
            Dict with dataset information
        """
        info = {
            "FEVER": {
                "description": "Fact Extraction and VERification dataset",
                "task": "Fact verification",
                "size": "~145K claims",
                "splits": ["train", "dev", "test"],
                "url": "https://fever.ai/",
                "recommended_for": "Factual consistency testing"
            },
            "HotpotQA": {
                "description": "Multi-hop question answering dataset",
                "task": "Question answering with reasoning chains",
                "size": "~113K questions",
                "splits": ["train", "dev_distractor", "dev_fullwiki"],
                "url": "https://hotpotqa.github.io/",
                "recommended_for": "Multi-hop reasoning, graph traversal"
            },
            "TruthfulQA": {
                "description": "Questions that test truthfulness",
                "task": "Measuring truthfulness in responses",
                "size": "~800 questions",
                "splits": ["main"],
                "url": "https://github.com/sylinrl/TruthfulQA",
                "recommended_for": "Hallucination detection"
            }
        }
        
        return info


def main():
    """Main function to run dataset downloads"""
    
    print("="*60)
    print("GRAPH-AUGMENTED RAG: DATASET DOWNLOADER")
    print("="*60)
    
    # Initialize downloader
    downloader = DatasetDownloader(base_dir="./data")
    
    # Show dataset info
    print("\nAvailable Datasets:")
    info = downloader.get_dataset_info()
    for name, details in info.items():
        print(f"\n{name}:")
        print(f"  - {details['description']}")
        print(f"  - Size: {details['size']}")
        print(f"  - Best for: {details['recommended_for']}")
    
    # Ask user what to download
    print("\n" + "="*60)
    print("What would you like to download?")
    print("="*60)
    print("1. FEVER only (recommended for factual consistency)")
    print("2. HotpotQA only (recommended for multi-hop reasoning)")
    print("3. TruthfulQA only (recommended for hallucination detection)")
    print("4. All datasets (~500 MB total)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        downloader.download_fever()
    elif choice == "2":
        downloader.download_hotpotqa()
    elif choice == "3":
        downloader.download_truthfulqa()
    elif choice == "4":
        downloader.download_all()
    elif choice == "5":
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
        return
    
    # Verify downloads
    print("\n")
    downloader.verify_datasets()
    
    print("\n" + "="*60)
    print("DONE! You can now use the datasets in your experiments.")
    print("="*60)
    print("\nExample usage:")
    print(">>> from dataset_loaders import DatasetLoader")
    print(">>> loader = DatasetLoader(cache_dir='./data')")
    print(">>> dataset = loader.load_fever(split='dev', max_samples=100)")


if __name__ == "__main__":
    # Install tqdm if not available
    try:
        import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        os.system("pip install tqdm")
        import tqdm
    
    main()