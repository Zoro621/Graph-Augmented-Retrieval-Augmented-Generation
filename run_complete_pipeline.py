"""
Complete Pipeline Runner for Graph-Augmented RAG Research
Run this script to execute the entire experiment pipeline

Usage:
    python run_complete_pipeline.py
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
from typing import Dict, List
from urllib.parse import quote


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from baseline_rag import BaselineRAG, RAGResponse
from graph_augmented_rag import GraphAugmentedRAG, GraphAugmentedResponse
from eval_framework import RAGEvaluator, EvaluationMetrics
from dataset_loaders import DatasetLoader, QueryDataset
from visualization_utils import RAGVisualizer

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()
class Config:
    """Experiment configuration"""
    
    # API Configuration - OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # OpenAI model
    
    # ===== COMMENTED OUT: Old OpenRouter configuration =====
    # OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # OpenRouter API key
    # MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-chat")
    # API_BASE = os.getenv("API_BASE")
    
    
    # Dataset Configuration
    DATASET_TYPE = "fever"  # Options: 'sample', 'fever', 'hotpotqa', 'truthfulqa'
    MAX_QUERIES = 20  # Number of queries to test
    
    # System Configuration
    USE_SPACY = False
    CONSISTENCY_THRESHOLD = 0.7
    K_RETRIEVE = 4
    
    # Output Configuration
    OUTPUT_DIR = "./results"
    SAVE_VISUALIZATIONS = True
    SAVE_DETAILED_LOGS = True
    SAVE_GRAPH_VISUALIZATIONS = True
    GRAPH_VIZ_MAX_NODES = 40
    USE_WIKI_SUMMARIES = True
    WIKI_CACHE_PATH = "./data/wiki_summary_cache.json"
    
    # Display Configuration
    VERBOSE = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text, char="="):
    """Print formatted header"""
    print(f"\n{char * 80}")
    print(f"{text.center(80)}")
    print(f"{char * 80}\n")


def print_section(text):
    """Print section divider"""
    print(f"\n{'─' * 80}")
    print(f"📌 {text}")
    print(f"{'─' * 80}")


def save_json(data, filepath):
    """Save data as JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {filepath}")


def print_progress(current, total, task="Processing"):
    """Print progress"""
    percentage = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r{task}: [{bar}] {percentage:.1f}% ({current}/{total})", end='')
    if current == total:
        print()  # New line when complete


class WikiSummaryFetcher:
    """Fetch and cache Wikipedia summaries for document titles."""

    def __init__(self, cache_path: Path, enabled: bool = True):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _fetch_summary(self, title: str) -> str:
        url_title = quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{url_title}"
        headers = {"User-Agent": "Agentic-GA-RAG/1.0"}
        try:
            response = requests.get(url, headers=headers, timeout=6)
            if response.status_code == 200:
                data = response.json()
                return data.get("extract", "")
        except requests.RequestException:
            return ""
        return ""

    def get_summary(self, title: str) -> str:
        if not self.enabled or not title:
            return ""
        key = title.strip()
        if not key:
            return ""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        summary = self._fetch_summary(key)
        self._cache[key] = summary
        self._save_cache()
        return summary


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ExperimentPipeline:
    """Complete experiment pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        self.start_time = datetime.now()
        self.documents = []
        self.document_metadatas = []
        
        # Create output directory
        Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        print_header("GRAPH-AUGMENTED RAG EXPERIMENT PIPELINE")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        print(f"  - Model: {config.MODEL_NAME}")
        print(f"  - Dataset: {config.DATASET_TYPE}")
        print(f"  - Max Queries: {config.MAX_QUERIES}")
        print(f"  - Output: {config.OUTPUT_DIR}")
    
    def step1_load_dataset(self):
        """Step 1: Load dataset"""
        print_section("STEP 1: Loading Dataset")
        
        loader = DatasetLoader(cache_dir="./data")
        
        if self.config.DATASET_TYPE == "sample":
            dataset = loader.create_sample_dataset()
        elif self.config.DATASET_TYPE == "fever":
            dataset = loader.load_fever(split="train", max_samples=self.config.MAX_QUERIES)
        elif self.config.DATASET_TYPE == "hotpotqa":
            dataset = loader.load_hotpotqa(split="dev", max_samples=self.config.MAX_QUERIES)
        elif self.config.DATASET_TYPE == "truthfulqa":
            dataset = loader.load_truthfulqa(max_samples=self.config.MAX_QUERIES)
        else:
            raise ValueError(f"Unknown dataset: {self.config.DATASET_TYPE}")
        
        # Limit queries
        dataset.queries = dataset.queries[:self.config.MAX_QUERIES]
        dataset.ground_truths = dataset.ground_truths[:self.config.MAX_QUERIES]
        dataset.contexts = dataset.contexts[:self.config.MAX_QUERIES]
        
        self.dataset = dataset
        self.documents, self.document_metadatas = self._build_document_corpus(dataset)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Queries: {len(dataset.queries)}")
        print(f"  - Documents: {len(self.documents)}")
        print(f"  - Sample query: {dataset.queries[0][:60]}...")
        
        return dataset

    def _build_document_corpus(self, dataset: QueryDataset):
        """Assemble textual documents from dataset contexts with optional wiki enrichment."""
        doc_titles: List[str] = []
        for context_list in dataset.contexts:
            for doc in context_list:
                title = (doc or "").strip()
                if title:
                    doc_titles.append(title)
        doc_titles = sorted(set(doc_titles))

        if not doc_titles:
            print("⚠️ No context documents found; using queries as fallback corpus")
            documents = dataset.queries
            metadatas = [{"source": f"query_{i}", "summary_used": False} for i in range(len(documents))]
            return documents, metadatas

        fetcher = WikiSummaryFetcher(
            cache_path=Path(self.config.WIKI_CACHE_PATH),
            enabled=self.config.USE_WIKI_SUMMARIES
        )

        documents = []
        metadatas = []
        summaries_available = 0
        for title in doc_titles:
            summary = fetcher.get_summary(title)
            if summary:
                text = summary
                summaries_available += 1
            else:
                text = title.replace("_", " ")
            documents.append(text)
            metadatas.append({
                "source": title,
                "summary_used": bool(summary)
            })

        print(
            f"  - Context titles: {len(doc_titles)} | Summaries downloaded: {summaries_available}"
        )

        return documents, metadatas
    
    def step2_run_baseline_rag(self):
        """Step 2: Run Baseline RAG"""
        print_section("STEP 2: Running Baseline RAG")
        
        # Initialize with OpenAI
        baseline_rag = BaselineRAG(
            api_key=Config.OPENAI_API_KEY,
            model_name=Config.MODEL_NAME,
            k_retrieve=Config.K_RETRIEVE
        )
        
        # ===== COMMENTED OUT: Old OpenRouter initialization =====
        # baseline_rag = BaselineRAG(
        #     api_key=Config.OPENROUTER_API_KEY,
        #     model_name=Config.MODEL_NAME,
        #     api_base="https://openrouter.ai/api/v1",
        #     k_retrieve=Config.K_RETRIEVE
        # )
        
        print("Initializing baseline RAG system...")
        baseline_rag.load_documents(self.documents, self.document_metadatas)
        baseline_rag.build_qa_chain()
        print("✓ Baseline RAG ready")
        
        # Run queries
        baseline_results = []
        baseline_times = []
        
        print(f"\nProcessing {len(self.dataset.queries)} queries...")
        
        for i, query in enumerate(self.dataset.queries, 1):
            print_progress(i, len(self.dataset.queries), "Baseline RAG")
            
            start_time = time.time()
            response = baseline_rag.query(query)
            elapsed = time.time() - start_time
            
            baseline_times.append(elapsed)
            baseline_results.append({
                "query": query,
                "answer": response.answer,
                "ground_truth": self.dataset.ground_truths[i-1],
                "retrieved_docs": response.retrieved_docs,
                "time": elapsed,
                "metadata": response.metadata
            })
        
        avg_time = np.mean(baseline_times)
        print(f"\n✓ Baseline RAG complete")
        print(f"  - Average time: {avg_time:.2f}s per query")
        print(f"  - Total time: {sum(baseline_times):.2f}s")
        
        self.baseline_results = baseline_results
        return baseline_results
    
    def step3_run_graph_augmented_rag(self):
        """Step 3: Run Graph-Augmented RAG"""
        print_section("STEP 3: Running Graph-Augmented RAG")
        
        # Initialize with OpenAI
        ga_rag = GraphAugmentedRAG(
            api_key=self.config.OPENAI_API_KEY,
            model_name=self.config.MODEL_NAME,
            use_spacy=self.config.USE_SPACY,
            consistency_threshold=self.config.CONSISTENCY_THRESHOLD,
            visualize_graphs=self.config.SAVE_GRAPH_VISUALIZATIONS,
            graph_viz_output_dir=str(Path(self.config.OUTPUT_DIR) / "figures" / "knowledge_graphs"),
            graph_viz_max_nodes=self.config.GRAPH_VIZ_MAX_NODES
        )
        
        # ===== COMMENTED OUT: Old OpenRouter initialization =====
        # ga_rag = GraphAugmentedRAG(
        #     api_key=self.config.OPENROUTER_API_KEY,
        #     model_name=self.config.MODEL_NAME,
        #     api_base="https://openrouter.ai/api/v1",
        #     use_spacy=self.config.USE_SPACY,
        #     consistency_threshold=self.config.CONSISTENCY_THRESHOLD
        # )
        
        print("Initializing Graph-Augmented RAG system...")
        ga_rag.load_documents(self.documents, self.document_metadatas)
        print("✓ Graph-Augmented RAG ready")
        
        # Run queries
        ga_rag_results = []
        ga_rag_times = []
        
        print(f"\nProcessing {len(self.dataset.queries)} queries...")
        
        for i, query in enumerate(self.dataset.queries, 1):
            print_progress(i, len(self.dataset.queries), "GA-RAG")
            
            start_time = time.time()
            response = ga_rag.query(query)
            elapsed = time.time() - start_time
            
            ga_rag_times.append(elapsed)
            ga_rag_results.append({
                "query": query,
                "answer": response.answer,
                "ground_truth": self.dataset.ground_truths[i-1],
                "retrieved_docs": response.retrieved_docs,
                "graph_triplets": response.graph_triplets,
                "ranked_triplets": response.ranked_triplets,
                "contradictions": response.contradictions,
                "inferences": response.inferences,
                "consistency_score": response.consistency_score,
                "factual_grounding_score": response.factual_grounding_score,
                "metadata": response.metadata,
                "time": elapsed
            })
        
        avg_time = np.mean(ga_rag_times)
        print(f"\n✓ Graph-Augmented RAG complete")
        print(f"  - Average time: {avg_time:.2f}s per query")
        print(f"  - Total time: {sum(ga_rag_times):.2f}s")
        print(f"  - Average consistency: {np.mean([r['consistency_score'] for r in ga_rag_results]):.3f}")
        
        self.ga_rag_results = ga_rag_results
        return ga_rag_results
    
    def step4_evaluate_systems(self):
        """Step 4: Evaluate and compare systems"""
        print_section("STEP 4: Evaluating Systems")
        
        evaluator = RAGEvaluator()
        
        # Evaluate baseline
        print("Evaluating Baseline RAG...")
        baseline_metrics = []
        for result in self.baseline_results:
            metrics = evaluator.evaluate_single_query(
                query=result["query"],
                answer=result["answer"],
                ground_truth=result["ground_truth"],
                retrieved_docs=result["retrieved_docs"],
                response_time=result["time"],
                factual_grounding=0.0,
                graph_coverage=0.0,
                context_precision=0.0
            )
            baseline_metrics.append(metrics)
        
        # Evaluate GA-RAG
        print("Evaluating Graph-Augmented RAG...")
        ga_rag_metrics = []
        for result in self.ga_rag_results:
            metrics = evaluator.evaluate_single_query(
                query=result["query"],
                answer=result["answer"],
                ground_truth=result["ground_truth"],
                retrieved_docs=result["retrieved_docs"],
                contradictions=result["contradictions"],
                graph_size=result["metadata"]["num_nodes"],
                num_triplets=len(result["graph_triplets"]),
                num_inferences=len(result["inferences"]),
                response_time=result["time"],
                factual_grounding=result.get("factual_grounding_score", 0.0),
                graph_coverage=result["metadata"].get("question_term_coverage", 0.0),
                context_precision=result["metadata"].get("context_filter_precision", 0.0)
            )
            ga_rag_metrics.append(metrics)
        
        # Compare systems
        comparison = evaluator.compare_systems(baseline_metrics, ga_rag_metrics)
        
        print("\n✓ Evaluation complete")
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"{'─' * 80}")
        
        for metric, improvement in comparison['improvements_pct'].items():
            baseline_val = comparison['baseline'][metric]
            ga_rag_val = comparison['graph_augmented'][metric]
            
            symbol = "📈" if improvement > 0 else "📉"
            print(f"{symbol} {metric}:")
            print(f"   Baseline: {baseline_val:.3f} | GA-RAG: {ga_rag_val:.3f} | Improvement: {improvement:+.1f}%")
        
        self.baseline_metrics = baseline_metrics
        self.ga_rag_metrics = ga_rag_metrics
        self.comparison = comparison
        
        return comparison
    
    def step5_generate_visualizations(self):
        """Step 5: Generate visualizations"""
        print_section("STEP 5: Generating Visualizations")
        
        if not self.config.SAVE_VISUALIZATIONS:
            print("Visualization generation disabled in config")
            return
        
        viz = RAGVisualizer()
        viz_dir = Path(self.config.OUTPUT_DIR) / "figures"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Metric comparison for each metric
        metrics_to_plot = [
            ("factual_accuracy", "Factual Accuracy"),
            ("logical_consistency", "Logical Consistency"),
            ("hallucination_rate", "Hallucination Rate"),
            ("response_coherence", "Response Coherence")
        ]
        
        query_labels = [f"Q{i+1}" for i in range(len(self.baseline_metrics))]
        
        for metric_key, metric_name in metrics_to_plot:
            print(f"Creating {metric_name} comparison plot...")
            baseline_vals = [getattr(m, metric_key) for m in self.baseline_metrics]
            ga_rag_vals = [getattr(m, metric_key) for m in self.ga_rag_metrics]
            
            viz.plot_metric_comparison(
                baseline_vals,
                ga_rag_vals,
                metric_name,
                query_labels,
                save_path=str(viz_dir / f"{metric_key}_comparison.png")
            )
        
        # 2. Aggregate comparison
        print("Creating aggregate comparison plot...")
        viz.plot_aggregate_comparison(
            self.comparison['baseline'],
            self.comparison['graph_augmented'],
            save_path=str(viz_dir / "aggregate_comparison.png")
        )
        
        # 3. Pipeline diagram
        print("Creating pipeline diagram...")
        viz.plot_pipeline_diagram(
            save_path=str(viz_dir / "pipeline_diagram.png")
        )
        
        # 4. Complete results summary
        print("Creating complete results summary...")
        viz.plot_results_summary(
            self.comparison,
            save_path=str(viz_dir / "results_summary.png")
        )
        
        print(f"\n✓ All visualizations saved to: {viz_dir}")
        
        return viz_dir
    
    def step6_export_results(self):
        """Step 6: Export results for paper"""
        print_section("STEP 6: Exporting Results")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save comparison results
        comparison_file = Path(self.config.OUTPUT_DIR) / f"comparison_results_{timestamp}.json"
        save_json(self.comparison, comparison_file)
        
        # 2. Save detailed results
        detailed_results = {
            "experiment_info": {
                "timestamp": timestamp,
                "dataset": self.config.DATASET_TYPE,
                "num_queries": len(self.dataset.queries),
                "model": self.config.MODEL_NAME,
                "configuration": {
                    "use_spacy": self.config.USE_SPACY,
                    "consistency_threshold": self.config.CONSISTENCY_THRESHOLD,
                    "k_retrieve": self.config.K_RETRIEVE
                }
            },
            "baseline_results": self.baseline_results,
            "ga_rag_results": self.ga_rag_results,
            "comparison": self.comparison
        }
        
        detailed_file = Path(self.config.OUTPUT_DIR) / f"detailed_results_{timestamp}.json"
        save_json(detailed_results, detailed_file)
        
        # 3. Create results table CSV
        table_data = []
        for i, (b_metric, ga_metric) in enumerate(zip(self.baseline_metrics, self.ga_rag_metrics), 1):
            table_data.append({
                "Query_ID": i,
                "Query": self.dataset.queries[i-1][:50] + "...",
                "Baseline_Accuracy": f"{b_metric.factual_accuracy:.3f}",
                "GA_RAG_Accuracy": f"{ga_metric.factual_accuracy:.3f}",
                "Baseline_Consistency": f"{b_metric.logical_consistency:.3f}",
                "GA_RAG_Consistency": f"{ga_metric.logical_consistency:.3f}",
                "Baseline_Hallucination": f"{b_metric.hallucination_rate:.3f}",
                "GA_RAG_Hallucination": f"{ga_metric.hallucination_rate:.3f}",
                "Baseline_Time": f"{b_metric.avg_response_time:.2f}s",
                "GA_RAG_Time": f"{ga_metric.avg_response_time:.2f}s"
            })
        
        df = pd.DataFrame(table_data)
        csv_file = Path(self.config.OUTPUT_DIR) / f"results_table_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved: {csv_file}")
        
        # 4. Create LaTeX table for paper
        latex_file = Path(self.config.OUTPUT_DIR) / f"results_table_{timestamp}.tex"
        with open(latex_file, 'w') as f:
            f.write(df.to_latex(index=False))
        print(f"✓ Saved: {latex_file}")
        
        # 5. Summary report
        summary_file = Path(self.config.OUTPUT_DIR) / f"experiment_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRAPH-AUGMENTED RAG EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Experiment Date: {timestamp}\n")
            f.write(f"Dataset: {self.config.DATASET_TYPE}\n")
            f.write(f"Number of Queries: {len(self.dataset.queries)}\n")
            f.write(f"Model: {self.config.MODEL_NAME}\n\n")
            
            f.write("AVERAGE METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Metric':<30} {'Baseline':<15} {'GA-RAG':<15} {'Improvement':<15}\n")
            f.write("-"*80 + "\n")
            
            for metric in self.comparison['baseline'].keys():
                baseline_val = self.comparison['baseline'][metric]
                ga_rag_val = self.comparison['graph_augmented'][metric]
                improvement = self.comparison['improvements_pct'][metric]
                f.write(f"{metric:<30} {baseline_val:<15.3f} {ga_rag_val:<15.3f} {improvement:+.1f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("-"*80 + "\n")
            
            summary_data = self.comparison['summary']
            f.write(f"• Accuracy Gain: {summary_data['accuracy_gain']:+.1f}%\n")
            f.write(f"• Consistency Gain: {summary_data['consistency_gain']:+.1f}%\n")
            f.write(f"• Hallucination Reduction: {summary_data['hallucination_reduction']:+.1f}%\n")
        
        print(f"✓ Saved: {summary_file}")
        
        print(f"\n✓ All results exported to: {self.config.OUTPUT_DIR}")
        
        return {
            "comparison": comparison_file,
            "detailed": detailed_file,
            "table_csv": csv_file,
            "table_latex": latex_file,
            "summary": summary_file
        }
    
    def run_complete_pipeline(self):
        """Run the complete experiment pipeline"""
        try:
            # Run all steps
            self.step1_load_dataset()
            self.step2_run_baseline_rag()
            self.step3_run_graph_augmented_rag()
            self.step4_evaluate_systems()
            self.step5_generate_visualizations()
            exported_files = self.step6_export_results()
            
            # Final summary
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            print_header("EXPERIMENT COMPLETE", "=")
            print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"\n✓ All results saved to: {self.config.OUTPUT_DIR}")
            print(f"\n📊 Key Results:")
            print(f"   - Queries Tested: {len(self.dataset.queries)}")
            print(f"   - Accuracy Improvement: {self.comparison['summary']['accuracy_gain']:+.1f}%")
            print(f"   - Consistency Improvement: {self.comparison['summary']['consistency_gain']:+.1f}%")
            print(f"   - Hallucination Reduction: {self.comparison['summary']['hallucination_reduction']:+.1f}%")
            
            print(f"\n📁 Generated Files:")
            for file_type, file_path in exported_files.items():
                print(f"   - {file_type}: {file_path}")
            
            print("\n🎉 SUCCESS! Your research data is ready for the paper.")
            print("="*80 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Check API key
    if Config. OPENAI_API_KEY == "your-openai-api-key-here":
        print("❌ ERROR: Please set your OpenAI API key in the Config class!")
        print("   Edit the OPENAI_API_KEY variable at the top of this script.")
        return
    
    # Create and run pipeline
    pipeline = ExperimentPipeline(Config)
    success = pipeline.run_complete_pipeline()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()