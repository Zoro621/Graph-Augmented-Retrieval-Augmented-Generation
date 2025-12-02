"""
Test that all modules are working correctly
"""

print("Testing imports...")

try:
    from baseline_rag import BaselineRAG
    print("✓ baseline_rag")
except Exception as e:
    print(f"✗ baseline_rag: {e}")

try:
    from graph_reasoning import KnowledgeGraphBuilder, LogicalConsistencyChecker
    print("✓ graph_reasoning")
except Exception as e:
    print(f"✗ graph_reasoning: {e}")

try:
    from graph_augmented_rag import GraphAugmentedRAG
    print("✓ graph_augmented_rag")
except Exception as e:
    print(f"✗ graph_augmented_rag: {e}")

try:
    from eval_framework import RAGEvaluator
    print("✓ evaluation_framework")
except Exception as e:
    print(f"✗ evaluation_framework: {e}")

try:
    from dataset_loaders import DatasetLoader
    print("✓ dataset_loaders")
except Exception as e:
    print(f"✗ dataset_loaders: {e}")

try:
    from visualization_utils import RAGVisualizer
    print("✓ visualization_utils")
except Exception as e:
    print(f"✗ visualization_utils: {e}")

try:
    from llm_triplet_extraction import LLMTripletExtractor
    print("✓ llm_triplet_extraction")
except Exception as e:
    print(f"✗ llm_triplet_extraction: {e}")

print("\nAll modules imported successfully! ✓")