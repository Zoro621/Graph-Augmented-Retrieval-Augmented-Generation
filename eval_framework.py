"""
Evaluation Framework for Graph-Augmented RAG Research
Implements metrics for comparison between baseline and GA-RAG
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    factual_accuracy: float
    logical_consistency: float
    hallucination_rate: float
    response_coherence: float
    graph_completeness: float
    avg_response_time: float
    factual_grounding: float = 0.0
    graph_coverage: float = 0.0
    context_precision: float = 0.0


class RAGEvaluator:
    """
    Comprehensive evaluation framework for RAG systems
    """
    
    def __init__(self):
        self.results = []
        self.baseline_results = []
        self.ga_rag_results = []
    
    def evaluate_factual_accuracy(
        self,
        answer: str,
        ground_truth: str,
        retrieved_docs: List[str]
    ) -> float:
        """
        Evaluate factual accuracy
        
        For FEVER dataset: checks if answer matches the claim verification label
        For other datasets: uses semantic similarity
        """
        answer_lower = answer.lower().strip()
        ground_truth_lower = ground_truth.lower().strip()
        
        # Handle FEVER-style labels (SUPPORTS, REFUTES, NOT ENOUGH INFO)
        fever_labels = ['supports', 'refutes', 'not enough info']
        
        if ground_truth_lower in fever_labels:
            # FIRST: Check if answer shows appropriate uncertainty for any label
            # Give partial credit when system is cautious with insufficient evidence
            uncertainty_indicators = [
                "i don't know", 'not sure', 'unclear', 'cannot determine',
                'no information', 'not mentioned', 'cannot confirm'
            ]
            if any(ind in answer_lower for ind in uncertainty_indicators):
                # Partial credit (0.3) for being appropriately cautious
                # Better than confidently wrong (0.0), worse than correct (1.0)
                return 0.3
            
            # Check if answer aligns with ground truth label
            if ground_truth_lower == 'supports':
                # Answer should confirm the claim
                positive_indicators = [
                    'yes', 'correct', 'true', 'indeed', 'is a', 'is an', 'was a',
                    'confirmed', 'accurate', 'supports', 'does'
                ]
                negative_indicators = [
                    "i don't know", 'no', 'incorrect', 'false', 'not', 'cannot'
                ]
                
                has_positive = any(ind in answer_lower for ind in positive_indicators)
                has_negative = any(ind in answer_lower for ind in negative_indicators)
                
                if has_positive and not has_negative:
                    return 1.0
                elif has_negative:
                    return 0.0
                else:
                    return 0.3  # Unclear
                    
            elif ground_truth_lower == 'refutes':
                # Answer should deny/contradict the claim
                negative_indicators = [
                    'no', 'incorrect', 'false', 'not', 'never', 'refutes',
                    'contradicts', 'wrong', 'isn\'t', 'wasn\'t', 'doesn\'t'
                ]
                positive_indicators = ['yes', 'correct', 'true', 'indeed']
                
                has_negative = any(ind in answer_lower for ind in negative_indicators)
                has_positive = any(ind in answer_lower for ind in positive_indicators)
                
                if has_negative and not has_positive:
                    return 1.0
                elif has_positive:
                    return 0.0
                else:
                    return 0.3
                    
            elif ground_truth_lower == 'not enough info':
                # Answer should indicate uncertainty
                uncertainty_indicators = [
                    "i don't know", 'not sure', 'unclear', 'cannot determine',
                    'no information', 'not mentioned', 'insufficient'
                ]
                
                has_uncertainty = any(ind in answer_lower for ind in uncertainty_indicators)
                
                if has_uncertainty:
                    return 1.0
                else:
                    # Partial credit if answer doesn't make strong claims
                    strong_claims = ['definitely', 'certainly', 'absolutely', 'yes', 'no']
                    has_strong = any(claim in answer_lower for claim in strong_claims)
                    return 0.0 if has_strong else 0.5
        
        else:
            # Traditional token overlap for non-FEVER datasets
            answer_tokens = set(answer_lower.split())
            truth_tokens = set(ground_truth_lower.split())
            
            if len(answer_tokens) == 0 or len(truth_tokens) == 0:
                return 0.0
            
            intersection = answer_tokens & truth_tokens
            precision = len(intersection) / len(answer_tokens)
            recall = len(intersection) / len(truth_tokens)
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
    
    def evaluate_logical_consistency(
        self,
        answer: str,
        contradictions: List[Dict],
        graph_size: int
    ) -> float:
        """
        Evaluate logical consistency
        
        Score = 1.0 if no contradictions, decreases with contradictions
        """
        if graph_size == 0:
            return 0.5  # Neutral score if no graph
        
        # Calculate contradiction rate
        contradiction_rate = len(contradictions) / max(1, graph_size)
        
        # Consistency score (inverse of contradiction rate)
        consistency = max(0.0, 1.0 - contradiction_rate)
        
        return consistency
    
    def evaluate_hallucination_rate(
        self,
        answer: str,
        retrieved_docs: List[str],
        strict: bool = False
    ) -> float:
        """
        Evaluate hallucination rate
        
        Measures proportion of answer content not supported by sources
        """
        # Strip meta-prefixes like [Low grounding 0.00] or [Graph coverage low (4 facts)]
        import re
        answer_cleaned = re.sub(r'^\[.*?\]\s*', '', answer)
        
        # Combine all retrieved documents
        source_text = " ".join(retrieved_docs).lower()
        source_tokens = set(source_text.split())
        
        # Extract answer tokens from cleaned answer
        answer_lower = answer_cleaned.lower()
        answer_tokens = answer_lower.split()
        
        # Check for common hedge phrases (indicates uncertainty)
        hedge_phrases = [
            "i don't know", "not sure", "unclear", "cannot determine",
            "no information", "not mentioned", "based on context"
        ]
        has_hedge = any(phrase in answer_lower for phrase in hedge_phrases)
        
        if has_hedge:
            return 0.0  # Model acknowledged uncertainty - not a hallucination
        
        # Common words to exclude (not indicative of hallucination)
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'of', 'in', 'to', 'for', 'with',
            'on', 'at', 'by', 'from', 'as', 'that', 'this', 'it', 'or', 'and',
            'but', 'not', 'yes', 'no', 'indeed', 'however', 'therefore'
        }
        
        # Count unsupported content words
        unsupported = 0
        content_words = 0
        
        for token in answer_tokens:
            # Skip punctuation and common words
            cleaned_token = token.strip('.,!?;:"\'')
            if len(cleaned_token) <= 2 or cleaned_token in common_words:
                continue
                
            content_words += 1
            
            # Check if token or its stem appears in source
            if cleaned_token not in source_tokens:
                # Check for partial matches (e.g., "running" in "run")
                stem_found = False
                for source_token in source_tokens:
                    if (cleaned_token.startswith(source_token) and len(source_token) > 3) or \
                       (source_token.startswith(cleaned_token) and len(cleaned_token) > 3):
                        stem_found = True
                        break
                
                if not stem_found:
                    unsupported += 1
        
        if content_words == 0:
            return 0.0
        
        hallucination_rate = unsupported / content_words
        
        return hallucination_rate
    
    def evaluate_response_coherence(self, answer: str) -> float:
        """
        Evaluate response coherence and quality
        
        Simple heuristics:
        - Length (not too short, not too long)
        - Sentence structure
        - Presence of answer indicators
        """
        # Length check
        word_count = len(answer.split())
        
        if word_count < 5:
            length_score = 0.3
        elif word_count > 200:
            length_score = 0.7
        else:
            length_score = 1.0
        
        # Check for complete sentences
        sentence_markers = ['.', '!', '?']
        has_sentences = any(marker in answer for marker in sentence_markers)
        sentence_score = 1.0 if has_sentences else 0.5
        
        # Check for answer indicators
        answer_indicators = ['is', 'are', 'was', 'were', 'the', 'a']
        has_indicators = any(ind in answer.lower() for ind in answer_indicators)
        indicator_score = 1.0 if has_indicators else 0.7
        
        # Combine scores
        coherence = (length_score + sentence_score + indicator_score) / 3
        
        return coherence
    
    def evaluate_graph_completeness(
        self,
        num_triplets: int,
        num_documents: int,
        num_inferences: int,
        query: str = ""
    ) -> float:
        """
        Evaluate how complete the knowledge graph is
        
        Measures density and relevance of extracted knowledge
        """
        if num_documents == 0:
            return 0.0
        
        # Average triplets per document (higher is better)
        triplets_per_doc = num_triplets / num_documents
        
        # Base score: expect 3-8 triplets per doc for good coverage
        # Scale: 0 triplets = 0.0, 3 triplets = 0.4, 8+ triplets = 1.0
        if triplets_per_doc >= 8:
            base_score = 1.0
        elif triplets_per_doc >= 3:
            base_score = 0.4 + (triplets_per_doc - 3) * 0.12  # Linear scaling
        else:
            base_score = triplets_per_doc * 0.133  # 0-3 range
        
        # Bonus for successful inferences (shows graph connectivity)
        if num_triplets > 0:
            inference_ratio = num_inferences / max(num_triplets, 1)
            inference_bonus = min(0.15, inference_ratio * 0.5)
        else:
            inference_bonus = 0.0
        
        completeness = min(1.0, base_score + inference_bonus)
        
        return completeness
    
    def evaluate_single_query(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        retrieved_docs: List[str],
        contradictions: List[Dict] = None,
        graph_size: int = 0,
        num_triplets: int = 0,
        num_inferences: int = 0,
        response_time: float = 0.0,
        factual_grounding: float = 0.0,
        graph_coverage: float = 0.0,
        context_precision: float = 0.0
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation for a single query
        """
        contradictions = contradictions or []
        
        metrics = EvaluationMetrics(
            factual_accuracy=self.evaluate_factual_accuracy(
                answer, ground_truth, retrieved_docs
            ),
            logical_consistency=self.evaluate_logical_consistency(
                answer, contradictions, graph_size
            ),
            hallucination_rate=self.evaluate_hallucination_rate(
                answer, retrieved_docs
            ),
            response_coherence=self.evaluate_response_coherence(answer),
            graph_completeness=self.evaluate_graph_completeness(
                num_triplets, len(retrieved_docs), num_inferences, query
            ),
            avg_response_time=response_time,
            factual_grounding=factual_grounding,
            graph_coverage=graph_coverage,
            context_precision=context_precision
        )

        return metrics
    
    def compare_systems(
        self,
        baseline_metrics: List[EvaluationMetrics],
        ga_rag_metrics: List[EvaluationMetrics]
    ) -> Dict[str, Any]:
        """
        Compare baseline RAG vs Graph-Augmented RAG and return averaged metrics.
        """
        def avg_metrics(metrics_list):
            return {
                "factual_accuracy": np.mean([m.factual_accuracy for m in metrics_list]),
                "logical_consistency": np.mean([m.logical_consistency for m in metrics_list]),
                "hallucination_rate": np.mean([m.hallucination_rate for m in metrics_list]),
                "response_coherence": np.mean([m.response_coherence for m in metrics_list]),
                "graph_completeness": np.mean([m.graph_completeness for m in metrics_list]),
                "avg_response_time": np.mean([m.avg_response_time for m in metrics_list]),
                "factual_grounding": np.mean([m.factual_grounding for m in metrics_list]),
                "graph_coverage": np.mean([m.graph_coverage for m in metrics_list]),
                "context_precision": np.mean([m.context_precision for m in metrics_list])
            }

        baseline_avg = avg_metrics(baseline_metrics)
        ga_rag_avg = avg_metrics(ga_rag_metrics)

        # Calculate improvements
        improvements = {
            metric: ((ga_rag_avg[metric] - baseline_avg[metric]) / baseline_avg[metric] * 100
                    if baseline_avg[metric] != 0 else 0)
            for metric in baseline_avg.keys()
        }

        # Special case: hallucination rate (lower is better)
        if baseline_avg["hallucination_rate"] != 0:
            improvements["hallucination_rate"] = (
                (baseline_avg["hallucination_rate"] - ga_rag_avg["hallucination_rate"]) /
                baseline_avg["hallucination_rate"] * 100
            )

        return {
            "baseline": baseline_avg,
            "graph_augmented": ga_rag_avg,
            "improvements_pct": improvements,
            "summary": {
                "accuracy_gain": improvements.get("factual_accuracy", 0),
                "consistency_gain": improvements.get("logical_consistency", 0),
                "hallucination_reduction": improvements.get("hallucination_rate", 0)
            }
        }
    
    def generate_results_table(
        self,
        baseline_metrics: List[EvaluationMetrics],
        ga_rag_metrics: List[EvaluationMetrics],
        query_names: List[str]
    ) -> pd.DataFrame:
        """
        Generate comparison table for paper/presentation
        """
        data = []
        
        for i, (b_metric, ga_metric, query) in enumerate(
            zip(baseline_metrics, ga_rag_metrics, query_names)
        ):
            data.append({
                "Query": query[:50] + "..." if len(query) > 50 else query,
                "Baseline_Accuracy": f"{b_metric.factual_accuracy:.3f}",
                "GA-RAG_Accuracy": f"{ga_metric.factual_accuracy:.3f}",
                "Baseline_Consistency": f"{b_metric.logical_consistency:.3f}",
                "GA-RAG_Consistency": f"{ga_metric.logical_consistency:.3f}",
                "Baseline_Hallucination": f"{b_metric.hallucination_rate:.3f}",
                "GA-RAG_Hallucination": f"{ga_metric.hallucination_rate:.3f}",
                "GA-RAG_Grounding": f"{ga_metric.factual_grounding:.3f}",
                "GA-RAG_Coverage": f"{ga_metric.graph_coverage:.3f}",
            })
        
        df = pd.DataFrame(data)
        return df

    def save_detailed_results(
        self,
        baseline_metrics: List[EvaluationMetrics],
        ga_rag_metrics: List[EvaluationMetrics],
        query_names: List[str],
        baseline_answers: List[str] = None,
        ga_answers: List[str] = None,
        ground_truths: List[str] = None,
        retrieved_docs_list: List[List[str]] = None,
        filename: str = "detailed_results.csv"
    ) -> str:
        """
        Save a per-query detailed CSV with metrics, answers and optional ground-truths.
        Returns the filename on success.
        """
        data = []
        for i, (b_metric, ga_metric, query) in enumerate(
            zip(baseline_metrics, ga_rag_metrics, query_names)
        ):
            row = {
                "query": query,
                "baseline_accuracy": b_metric.factual_accuracy,
                "ga_rag_accuracy": ga_metric.factual_accuracy,
                "baseline_consistency": b_metric.logical_consistency,
                "ga_rag_consistency": ga_metric.logical_consistency,
                "baseline_hallucination": b_metric.hallucination_rate,
                "ga_rag_hallucination": ga_metric.hallucination_rate,
                "ga_rag_grounding": ga_metric.factual_grounding,
                "ga_rag_coverage": ga_metric.graph_coverage,
            }

            if baseline_answers:
                row["baseline_answer"] = baseline_answers[i]
            if ga_answers:
                row["ga_rag_answer"] = ga_answers[i]
            if ground_truths:
                row["ground_truth"] = ground_truths[i]
            if retrieved_docs_list:
                row["retrieved_doc_count"] = len(retrieved_docs_list[i]) if retrieved_docs_list[i] else 0

            data.append(row)

        df = pd.DataFrame(data)
        try:
            df.to_csv(filename, index=False)
            print(f"✓ Saved detailed CSV to {filename}")
            return filename
        except Exception as exc:
            print(f"⚠️ Failed to save CSV {filename}: {exc}")
            raise
    
    def save_results(
        self,
        comparison: Dict[str, Any],
        filename: str = "evaluation_results.json"
    ):
        """Save evaluation results to file"""
        with open(filename, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✓ Results saved to {filename}")


# Example evaluation workflow
if __name__ == "__main__":
    # Simulated evaluation data
    evaluator = RAGEvaluator()
    
    # Example test cases
    test_cases = [
        {
            "query": "When was Einstein born?",
            "ground_truth": "Einstein was born in 1879 in Germany.",
            "baseline_answer": "Einstein was born in 1879.",
            "ga_rag_answer": "Einstein was born in 1879 in Germany.",
            "retrieved_docs": ["Albert Einstein was born in Germany in 1879."],
            "contradictions": [],
            "graph_size": 5,
            "num_triplets": 3,
            "num_inferences": 1
        },
        {
            "query": "What prize did Einstein receive?",
            "ground_truth": "Einstein received the Nobel Prize in Physics in 1921.",
            "baseline_answer": "Einstein received the Nobel Prize in 1921 for his theory of relativity.",
            "ga_rag_answer": "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.",
            "retrieved_docs": ["Einstein received the Nobel Prize in Physics in 1921 for his work on the photoelectric effect."],
            "contradictions": [],
            "graph_size": 8,
            "num_triplets": 5,
            "num_inferences": 2
        }
    ]
    
    baseline_metrics = []
    ga_rag_metrics = []
    query_names = []
    
    print("\n" + "="*80)
    print("EVALUATION FRAMEWORK - SAMPLE RUN")
    print("="*80)
    
    for test in test_cases:
        query_names.append(test["query"])
        
        # Evaluate baseline
        baseline_metric = evaluator.evaluate_single_query(
            query=test["query"],
            answer=test["baseline_answer"],
            ground_truth=test["ground_truth"],
            retrieved_docs=test["retrieved_docs"]
        )
        baseline_metrics.append(baseline_metric)
        
        # Evaluate GA-RAG
        ga_rag_metric = evaluator.evaluate_single_query(
            query=test["query"],
            answer=test["ga_rag_answer"],
            ground_truth=test["ground_truth"],
            retrieved_docs=test["retrieved_docs"],
            contradictions=test["contradictions"],
            graph_size=test["graph_size"],
            num_triplets=test["num_triplets"],
            num_inferences=test["num_inferences"]
        )
        ga_rag_metrics.append(ga_rag_metric)
    
    # Compare systems
    comparison = evaluator.compare_systems(baseline_metrics, ga_rag_metrics)
    
    print("\n📊 COMPARISON RESULTS:")
    print(json.dumps(comparison["summary"], indent=2))
    
    # Generate results table
    results_table = evaluator.generate_results_table(
        baseline_metrics, ga_rag_metrics, query_names
    )
    
    print("\n📋 RESULTS TABLE:")
    print(results_table.to_string(index=False))
    
    # Save results
    evaluator.save_results(comparison, "sample_evaluation_results.json")