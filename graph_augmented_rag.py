# """
# Graph-Augmented RAG System - Complete Integration
# Combines retrieval, graph reasoning, and generation
# """

# import json
# from typing import List, Dict, Any
# from dataclasses import dataclass
# import networkx as nx

# # Import from previous modules
# from baseline_rag import BaselineRAG, RAGResponse
# from graph_reasoning import KnowledgeGraphBuilder, LogicalConsistencyChecker, Triplet


# @dataclass
# class GraphAugmentedResponse:
#     """Enhanced response with graph reasoning information"""
#     query: str
#     answer: str
#     retrieved_docs: List[str]
#     graph_triplets: List[Dict]
#     contradictions: List[Dict]
#     inferences: List[Dict]
#     consistency_score: float
#     metadata: Dict[str, Any]


# class GraphAugmentedRAG:
#     """
#     Main Graph-Augmented RAG System
    
#     Pipeline:
#     1. Retrieve relevant documents (using BaselineRAG)
#     2. Extract knowledge triplets from retrieved docs
#     3. Build knowledge graph
#     4. Apply logical consistency checks
#     5. Generate answer using validated graph context
#     """
    
#     def __init__(
#         self,
#         api_key: str,
#         model_name: str = "deepseek/deepseek-chat",  # or any OpenRouter model
#         api_base: str = "https://openrouter.ai/api/v1",
#         use_spacy: bool = False,
#         consistency_threshold: float = 0.7
#     ):
#         """
#         Initialize Graph-Augmented RAG
        
#         Args:
#             api_key: OpenRouter API key
#             model_name: LLM model for generation
#             api_base: OpenRouter API base URL
#             use_spacy: Whether to use spaCy for NER
#             consistency_threshold: Minimum consistency score to accept answer
#         """
        
#         self.api_key = api_key
#         self.api_base = api_base
#         self.model_name = model_name
#         self.use_spacy = use_spacy
#         self.consistency_threshold = consistency_threshold
#         # Set environment variables for LangChain
#         import os
#         os.environ["OPENAI_API_KEY"] = api_key
#         os.environ["OPENAI_API_BASE"] = api_base
        
#         # Import BaselineRAG and initialize it with OpenRouter-compatible API
#         from baseline_rag import BaselineRAG
#         self.base_rag = BaselineRAG(
#             api_key=api_key,
#             model_name=model_name,
#             api_base=api_base
#         )
        
#         self.model_name = model_name
#         self.use_spacy = use_spacy
#         self.consistency_threshold = consistency_threshold
        
#         print("✓ Graph-Augmented RAG initialized (OpenRouter compatible)")
    
    
    
#     # def __init__(
#     #     self,
#     #     openai_api_key: str,
#     #     model_name: str = "gpt-3.5-turbo",
#     #     use_spacy: bool = True,
#     #     consistency_threshold: float = 0.7
#     # ):
#     #     """
#     #     Initialize Graph-Augmented RAG
        
#     #     Args:
#     #         openai_api_key: OpenAI API key
#     #         model_name: LLM model for generation
#     #         use_spacy: Whether to use spaCy for NER
#     #         consistency_threshold: Minimum consistency score to accept answer
#     #     """
#     #     # Initialize baseline RAG (retrieval + generation)
#     #     from baseline_rag import BaselineRAG
#     #     self.base_rag = BaselineRAG(
#     #         openai_api_key=openai_api_key,
#     #         model_name=model_name
#     #     )
        
#     #     self.model_name = model_name
#     #     self.use_spacy = use_spacy
#     #     self.consistency_threshold = consistency_threshold
        
#     #     print("✓ Graph-Augmented RAG initialized")
    
#     def load_documents(self, documents: List[str], metadatas: List[Dict] = None):
#         """Load documents into the system"""
#         self.base_rag.load_documents(documents, metadatas)
#         self.base_rag.build_qa_chain()
        
#     def _build_graph_from_retrieved(self, retrieved_docs: List[str]) -> tuple:
#         """
#         Build knowledge graph from retrieved documents
#         NOW USES LLM EXTRACTION!
#         """
#         # Import LLM extractor
#         from llm_triplet_extraction import LLMTripletExtractor
        
#         # Initialize LLM extractor (use same API as main system)
#         llm_extractor = LLMTripletExtractor(
#             api_key=self.api_key,  # Your DeepSeek key
#             model=self.model_name,         # deepseek/deepseek-chat
#             temperature=0.0
#         )
        
#         # Override base URL for OpenRouter
#         if hasattr(self, 'api_base') and self.api_base:
#             llm_extractor.client.base_url = self.api_base
        
#         print(f"\n🤖 Using LLM extraction with {self.model_name}...")
        
#         # Extract triplets using LLM (NOT SpaCy!)
#         all_triplets = []
#         for i, doc in enumerate(retrieved_docs[:3]):  # Limit to 3 docs to save cost
#             if len(doc.strip()) < 20:
#                 continue
            
#             print(f"  Extracting from doc {i+1}/3...", end='\r')
            
#             # Use LLM to extract triplets
#             triplets = llm_extractor.extract_triplets_single(
#                 text=doc,
#                 max_triplets=5  # Limit to 5 per doc
#             )
            
#             # Filter by confidence
#             triplets = [t for t in triplets if t.confidence > 0.6]
#             all_triplets.extend(triplets)
        
#         print(f"\n✓ Extracted {len(all_triplets)} triplets using LLM")
        
#         # Build graph from LLM-extracted triplets
#         builder = KnowledgeGraphBuilder(use_spacy=False)  # Don't use SpaCy
#         builder.add_triplets(all_triplets)
        
#         # Run consistency checks (same as before)
#         checker = LogicalConsistencyChecker(builder.graph)
#         checker.check_mutual_exclusivity({
#             "born": ["died"],
#             "is_alive": ["died", "passed_away"],
#             "created": ["destroyed"]
#         })
#         checker.check_transitivity(["is_part_of", "located_in", "member_of"])
#         checker.check_temporal_consistency()
        
#         return builder, checker
    
#     # def _build_graph_from_retrieved(self, retrieved_docs: List[str]) -> tuple:
#     #     """
#     #     Build knowledge graph from retrieved documents
        
#     #     Returns:
#     #         (KnowledgeGraphBuilder, LogicalConsistencyChecker)
#     #     """
#     #     from graph_reasoning import KnowledgeGraphBuilder, LogicalConsistencyChecker
        
#     #     # Build graph
#     #     builder = KnowledgeGraphBuilder(use_spacy=self.use_spacy)
#     #     builder.build_from_documents(retrieved_docs)
        
#     #     # Run consistency checks
#     #     checker = LogicalConsistencyChecker(builder.graph)
        
#     #     # Apply various consistency checks
#     #     checker.check_mutual_exclusivity({
#     #         "born": ["died"],
#     #         "is_alive": ["died", "passed_away"],
#     #         "created": ["destroyed"]
#     #     })
        
#     #     checker.check_transitivity(["is_part_of", "located_in", "member_of"])
#     #     checker.check_temporal_consistency()
        
#     #     return builder, checker
    
#     def _graph_to_context(
#         self,
#         graph: nx.DiGraph,
#         query: str,
#         max_facts: int = 10
#     ) -> str:
#         """
#         Convert validated graph to natural language context
        
#         Args:
#             graph: Knowledge graph
#             query: Original query (for relevance ranking)
#             max_facts: Maximum number of facts to include
            
#         Returns:
#             Natural language context string
#         """
#         context_parts = []
        
#         # Extract edges as facts
#         for u, v, data in graph.edges(data=True):
#             relation = data.get("relation", "related_to")
#             fact = f"{u} {relation} {v}."
#             context_parts.append(fact)
        
#         # Limit to max_facts (could add relevance scoring here)
#         context_parts = context_parts[:max_facts]
        
#         return " ".join(context_parts)
    
#     def _calculate_consistency_score(
#         self,
#         total_triplets: int,
#         contradictions: int,
#         inferences: int
#     ) -> float:
#         """
#         Calculate overall consistency score
        
#         Score = 1.0 - (contradictions / total_triplets) + (inferences_bonus)
#         """
#         if total_triplets == 0:
#             return 0.0
        
#         # Penalize contradictions
#         contradiction_penalty = contradictions / total_triplets
        
#         # Reward successful inferences (small bonus)
#         inference_bonus = min(0.1, inferences / (total_triplets * 10))
        
#         score = max(0.0, 1.0 - contradiction_penalty + inference_bonus)
#         return min(1.0, score)
    
#     def query(self, question: str) -> GraphAugmentedResponse:
#         """
#         Query the Graph-Augmented RAG system
        
#         Args:
#             question: User query
            
#         Returns:
#             GraphAugmentedResponse with full reasoning trace
#         """
#         # Step 1: Retrieve documents using base RAG
#         base_response = self.base_rag.query(question)
#         retrieved_docs = base_response.retrieved_docs
        
#         print(f"📚 Retrieved {len(retrieved_docs)} documents")
        
#         # Step 2: Build knowledge graph from retrieved docs
#         builder, checker = self._build_graph_from_retrieved(retrieved_docs)
        
#         print(f"🕸️  Built graph with {builder.graph.number_of_nodes()} nodes")
        
#         # Step 3: Get validated subgraph (contradictions removed)
#         validated_graph = checker.get_validated_subgraph()
        
#         # Step 4: Calculate consistency score
#         consistency_score = self._calculate_consistency_score(
#             total_triplets=len(builder.triplets),
#             contradictions=len(checker.contradictions),
#             inferences=len(checker.inferences)
#         )
        
#         print(f"✅ Consistency score: {consistency_score:.2f}")
        
#         # Step 5: Generate answer using validated graph context
#         graph_context = self._graph_to_context(validated_graph, question)
        
#         # Create enhanced prompt with graph context
#         if consistency_score >= self.consistency_threshold:
#             enhanced_question = f"""Based on the following validated knowledge:
# {graph_context}

# Question: {question}

# Provide a factual answer based only on the validated knowledge above."""
            
#             # Get final answer
#             final_response = self.base_rag.query(enhanced_question)
#             answer = final_response.answer
#         else:
#             # Low consistency - flag to user
#             answer = f"[Low consistency detected: {consistency_score:.2f}] " + base_response.answer
        
#         # Step 6: Create comprehensive response
#         response = GraphAugmentedResponse(
#             query=question,
#             answer=answer,
#             retrieved_docs=retrieved_docs,
#             graph_triplets=[
#                 {
#                     "subject": t.subject,
#                     "relation": t.relation,
#                     "object": t.object
#                 }
#                 for t in list(builder.triplets)[:20]  # Limit for display
#             ],
#             contradictions=checker.contradictions,
#             inferences=[
#                 {"subject": inf[0], "relation": inf[1], "object": inf[2]}
#                 for inf in checker.inferences
#             ],
#             consistency_score=consistency_score,
#             metadata={
#                 "num_nodes": validated_graph.number_of_nodes(),
#                 "num_edges": validated_graph.number_of_edges(),
#                 "num_contradictions": len(checker.contradictions),
#                 "num_inferences": len(checker.inferences),
#                 "model": self.model_name
#             }
#         )
        
#         return response
    
#     def compare_with_baseline(self, question: str) -> Dict[str, Any]:
#         """
#         Compare Graph-Augmented RAG with baseline
        
#         Returns comparison metrics
#         """
#         # Get baseline response
#         baseline_response = self.base_rag.query(question)
#         baseline_eval = self.base_rag.evaluate_hallucination(baseline_response)
        
#         # Get graph-augmented response
#         ga_response = self.query(question)
        
#         return {
#             "question": question,
#             "baseline": {
#                 "answer": baseline_response.answer,
#                 "evaluation": baseline_eval
#             },
#             "graph_augmented": {
#                 "answer": ga_response.answer,
#                 "consistency_score": ga_response.consistency_score,
#                 "contradictions": len(ga_response.contradictions),
#                 "graph_size": ga_response.metadata["num_nodes"]
#             },
#             "improvement": {
#                 "consistency_gain": ga_response.consistency_score - baseline_eval["word_overlap_ratio"],
#                 "logical_validation": len(ga_response.contradictions) == 0
#             }
#         }


# # Example usage and evaluation
# if __name__ == "__main__":
#     # Sample knowledge base
#     documents = [
#         "Albert Einstein was born in Germany in 1879. He was a theoretical physicist.",
#         "Einstein developed the theory of relativity in 1905. This theory revolutionized physics.",
#         "In 1921, Einstein received the Nobel Prize in Physics for his work on the photoelectric effect.",
#         "Einstein moved to the United States in 1933 and became a professor at Princeton University.",
#         "Einstein died in 1955 at the age of 76 in Princeton, New Jersey.",
#         "The theory of relativity consists of special relativity and general relativity.",
#         "Special relativity introduced the famous equation E=mc²."
#     ]
    
#     # Initialize system
#     API_KEY = "your-openai-api-key-here"
    
#     ga_rag = GraphAugmentedRAG(
#     api_key=API_KEY,  # your OpenRouter API key
#     model_name="deepseek/deepseek-chat",  # or any other OpenRouter-supported model
#     api_base="https://openrouter.ai/api/v1",
#     use_spacy=False,
#     consistency_threshold=0.7
# )

    
#     # ga_rag = GraphAugmentedRAG(
#     #     openai_api_key=API_KEY,
#     #     model_name="gpt-3.5-turbo",
#     #     use_spacy=True,
#     #     consistency_threshold=0.7
#     # )
    
#     # Load documents
#     ga_rag.load_documents(documents)
    
#     # Test queries
#     test_queries = [
#         "When and where was Einstein born?",
#         "What did Einstein develop?",
#         "When did Einstein receive the Nobel Prize?",
#         "What is E=mc²?"
#     ]
    
#     print("\n" + "="*80)
#     print("GRAPH-AUGMENTED RAG SYSTEM - EVALUATION")
#     print("="*80)
    
#     for query in test_queries:
#         print(f"\n{'='*80}")
#         print(f"📝 QUERY: {query}")
#         print(f"{'='*80}")
        
#         # Get comparison
#         comparison = ga_rag.compare_with_baseline(query)
        
#         print(f"\n🔵 BASELINE RAG:")
#         print(f"   Answer: {comparison['baseline']['answer']}")
#         print(f"   Evaluation: {json.dumps(comparison['baseline']['evaluation'], indent=2)}")
        
#         print(f"\n🟢 GRAPH-AUGMENTED RAG:")
#         print(f"   Answer: {comparison['graph_augmented']['answer']}")
#         print(f"   Consistency Score: {comparison['graph_augmented']['consistency_score']:.2f}")
#         print(f"   Contradictions Found: {comparison['graph_augmented']['contradictions']}")
#         print(f"   Graph Size: {comparison['graph_augmented']['graph_size']} nodes")
        
#         print(f"\n📊 IMPROVEMENT:")
#         print(f"   Consistency Gain: {comparison['improvement']['consistency_gain']:.2f}")
#         print(f"   Logically Validated: {comparison['improvement']['logical_validation']}")
        
#         print("-" * 80)

# """
# Enhanced Graph-Augmented RAG System - Improved Version
# Addresses triplet coverage, graph context ranking, consistency scoring, and more
# """

# import json
# import numpy as np
# from typing import List, Dict, Any, Tuple, Set
# from dataclasses import dataclass, field
# from collections import defaultdict
# import networkx as nx
# from difflib import SequenceMatcher

# # Import from previous modules
# from baseline_rag import BaselineRAG, RAGResponse
# from graph_reasoning import KnowledgeGraphBuilder, LogicalConsistencyChecker, Triplet


# @dataclass
# class GraphAugmentedResponse:
#     """Enhanced response with graph reasoning information"""
#     query: str
#     answer: str
#     retrieved_docs: List[str]
#     graph_triplets: List[Dict]
#     ranked_triplets: List[Dict]  # NEW: ranked by relevance
#     contradictions: List[Dict]
#     inferences: List[Dict]
#     consistency_score: float
#     factual_grounding_score: float  # NEW: how well-grounded in graph
#     metadata: Dict[str, Any]


# class TripletNormalizer:
#     """Deduplicates and normalizes triplets"""
    
#     @staticmethod
#     def normalize_text(text: str) -> str:
#         """Normalize text for comparison"""
#         return text.lower().strip().replace("  ", " ")
    
#     @staticmethod
#     def text_similarity(text1: str, text2: str) -> float:
#         """Calculate similarity between two texts (0.0 to 1.0)"""
#         norm1 = TripletNormalizer.normalize_text(text1)
#         norm2 = TripletNormalizer.normalize_text(text2)
        
#         if norm1 == norm2:
#             return 1.0
        
#         return SequenceMatcher(None, norm1, norm2).ratio()
    
#     @staticmethod
#     def deduplicate_triplets(triplets: List[Triplet], similarity_threshold: float = 0.85) -> List[Triplet]:
#         """
#         Remove duplicate triplets using similarity matching
#         Keeps the one with highest confidence
#         """
#         if not triplets:
#             return []
        
#         # Sort by confidence descending
#         sorted_triplets = sorted(triplets, key=lambda t: t.confidence, reverse=True)
        
#         deduplicated = []
#         seen_normalized = []
        
#         for triplet in sorted_triplets:
#             norm_key = (
#                 TripletNormalizer.normalize_text(triplet.subject),
#                 TripletNormalizer.normalize_text(triplet.relation),
#                 TripletNormalizer.normalize_text(triplet.object)
#             )
            
#             is_duplicate = False
#             for seen_key in seen_normalized:
#                 # Check each component
#                 subject_sim = TripletNormalizer.text_similarity(norm_key[0], seen_key[0])
#                 relation_sim = TripletNormalizer.text_similarity(norm_key[1], seen_key[1])
#                 object_sim = TripletNormalizer.text_similarity(norm_key[2], seen_key[2])
                
#                 avg_sim = (subject_sim + relation_sim + object_sim) / 3
                
#                 if avg_sim >= similarity_threshold:
#                     is_duplicate = True
#                     break
            
#             if not is_duplicate:
#                 deduplicated.append(triplet)
#                 seen_normalized.append(norm_key)
        
#         return deduplicated


# class TripletRanker:
#     """Ranks triplets by relevance to query"""
    
#     @staticmethod
#     def calculate_relevance_score(
#         triplet: Triplet,
#         query: str,
#         retrieval_score: float = 1.0,
#         mention_count: int = 1
#     ) -> float:
#         """
#         Calculate combined relevance score
#         Formula: confidence × (1 + mention_count*0.1) × retrieval_score × query_relevance
#         """
#         # Query relevance: how much query overlaps with triplet
#         query_lower = query.lower()
#         triplet_text = f"{triplet.subject} {triplet.relation} {triplet.object}".lower()
        
#         query_words = set(query_lower.split())
#         triplet_words = set(triplet_text.split())
        
#         overlap = len(query_words & triplet_words)
#         query_relevance = 1.0 + (overlap * 0.2)
        
#         # Combine scores
#         mention_factor = 1.0 + (mention_count * 0.1)
        
#         score = (
#             triplet.confidence * 
#             mention_factor * 
#             retrieval_score * 
#             query_relevance
#         )
        
#         return score
    
#     @staticmethod
#     def rank_triplets(
#         triplets: List[Triplet],
#         query: str,
#         retrieval_scores: Dict[str, float] = None,
#         top_k: int = None
#     ) -> List[Tuple[Triplet, float]]:
#         """
#         Rank triplets by relevance
        
#         Returns:
#             List of (triplet, score) tuples sorted by score descending
#         """
#         if retrieval_scores is None:
#             retrieval_scores = defaultdict(lambda: 1.0)
        
#         # Count mentions (triplets with same subject/object)
#         mention_counts = defaultdict(int)
#         for t in triplets:
#             key = (t.subject, t.relation, t.object)
#             mention_counts[key] += 1
        
#         # Score each triplet
#         scored = []
#         for triplet in triplets:
#             key = (triplet.subject, triplet.relation, triplet.object)
#             mention_count = mention_counts[key]
            
#             # Get retrieval score (default 1.0 if not provided)
#             retrieval_score = retrieval_scores.get(triplet.source_text, 1.0)
            
#             score = TripletRanker.calculate_relevance_score(
#                 triplet=triplet,
#                 query=query,
#                 retrieval_score=retrieval_score,
#                 mention_count=mention_count
#             )
            
#             scored.append((triplet, score))
        
#         # Sort by score descending
#         ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        
#         if top_k:
#             ranked = ranked[:top_k]
        
#         return ranked


# class ConsistencyChecker:
#     """Enhanced consistency checking with more comprehensive rules"""
    
#     # Expanded mutual exclusivity rules
#     MUTUAL_EXCLUSIVITY_RULES = {
#         "born": ["died", "passed_away", "was_assassinated"],
#         "is_alive": ["died", "passed_away", "was_assassinated"],
#         "created": ["destroyed", "demolished"],
#         "is_author_of": ["is_translator_of"],  # Same person usually doesn't both author and translate
#         "located_in": ["located_not_in"],
#         "member_of": ["not_member_of"],
#         "is_before": ["is_after"],
#         "is_contemporary_with": ["is_before", "is_after"],
#     }
    
#     # Transitivity rules
#     TRANSITIVITY_RULES = [
#         ("is_part_of", "is_part_of"),
#         ("located_in", "located_in"),
#         ("member_of", "member_of"),
#         ("is_ancestor_of", "is_ancestor_of"),
#         ("is_supervisor_of", "is_supervisor_of"),
#     ]
    
#     def __init__(self, graph: nx.DiGraph):
#         self.graph = graph
#         self.contradictions = []
#         self.inferences = []
#         self.confidence_scores = {}
    
#     def check_comprehensive_mutual_exclusivity(self):
#         """Check mutual exclusivity with logging"""
#         violations = []
        
#         for source_relation, conflicting_relations in self.MUTUAL_EXCLUSIVITY_RULES.items():
#             # Find nodes with source_relation edges
#             for u, v, data in self.graph.edges(data=True):
#                 if data.get("relation") == source_relation:
#                     # Check if any conflicting edge exists
#                     for conflict_rel in conflicting_relations:
#                         if self.graph.has_edge(u, v) and self.graph[u][v].get("relation") == conflict_rel:
#                             violations.append({
#                                 "type": "mutual_exclusivity",
#                                 "subject": u,
#                                 "object": v,
#                                 "conflicting_relations": [source_relation, conflict_rel]
#                             })
        
#         self.contradictions.extend(violations)
#         return violations
    
#     def check_comprehensive_transitivity(self):
#         """Check transitivity with inference generation"""
#         new_inferences = []
        
#         for rel1, rel2 in self.TRANSITIVITY_RULES:
#             # Find chains: A --rel1--> B --rel2--> C
#             for u in self.graph.nodes():
#                 for v in self.graph.neighbors(u):
#                     if self.graph[u][v].get("relation") == rel1:
#                         for w in self.graph.neighbors(v):
#                             if self.graph[v][w].get("relation") == rel2:
#                                 # Check if direct edge exists
#                                 if not self.graph.has_edge(u, w):
#                                     new_inferences.append((u, rel1, w))
        
#         self.inferences.extend(new_inferences)
#         return new_inferences
    
#     def check_temporal_consistency(self):
#         """Detect temporal violations"""
#         violations = []
        
#         # Find temporal edges
#         temporal_edges = []
#         for u, v, data in self.graph.edges(data=True):
#             rel = data.get("relation", "")
#             if any(t in rel.lower() for t in ["before", "after", "during", "year"]):
#                 temporal_edges.append((u, v, rel))
        
#         # Simple check: if A is_before B and B is_before C, then A should be_before C
#         # (simplified - would need actual temporal values in production)
        
#         return violations


# class GraphAugmentedRAG:
#     """
#     Enhanced Graph-Augmented RAG System with improved coverage and consistency
    
#     Key Improvements:
#     1. Increased triplet extraction (5-10 docs, 10-15 triplets per doc)
#     2. Triplet deduplication using similarity matching
#     3. Ranked triplet injection based on query relevance
#     4. Enhanced consistency checking with expanded rules
#     5. Factual grounding score
#     6. Better debug output
#     """
    
#     def __init__(
#         self,
#         api_key: str,
#         model_name: str = "deepseek/deepseek-chat",
#         api_base: str = "https://openrouter.ai/api/v1",
#         use_spacy: bool = False,
#         consistency_threshold: float = 0.7,
#         max_docs: int = 10,  # IMPROVED: extract from more docs
#         max_triplets_per_doc: int = 15,  # IMPROVED: more triplets per doc
#         max_context_triplets: int = 25,  # IMPROVED: inject more facts
#         triplet_confidence_threshold: float = 0.5,  # IMPROVED: lower threshold
#         similarity_threshold: float = 0.85,  # For deduplication
#     ):
#         """Initialize Enhanced Graph-Augmented RAG"""
        
#         self.api_key = api_key
#         self.api_base = api_base
#         self.model_name = model_name
#         self.use_spacy = use_spacy
#         self.consistency_threshold = consistency_threshold
#         self.max_docs = max_docs
#         self.max_triplets_per_doc = max_triplets_per_doc
#         self.max_context_triplets = max_context_triplets
#         self.triplet_confidence_threshold = triplet_confidence_threshold
#         self.similarity_threshold = similarity_threshold
        
#         import os
#         os.environ["OPENAI_API_KEY"] = api_key
#         os.environ["OPENAI_API_BASE"] = api_base
        
#         from baseline_rag import BaselineRAG
#         self.base_rag = BaselineRAG(
#             api_key=api_key,
#             model_name=model_name,
#             api_base=api_base
#         )
        
#         print("✓ Enhanced Graph-Augmented RAG initialized")
#         print(f"  - Max docs per query: {max_docs}")
#         print(f"  - Max triplets per doc: {max_triplets_per_doc}")
#         print(f"  - Max context triplets: {max_context_triplets}")
    
#     def load_documents(self, documents: List[str], metadatas: List[Dict] = None):
#         """Load documents into the system"""
#         self.base_rag.load_documents(documents, metadatas)
#         self.base_rag.build_qa_chain()
    
#     def _build_graph_from_retrieved(self, retrieved_docs: List[str], query: str) -> Tuple:
#         """
#         Build knowledge graph from retrieved documents with improved coverage
        
#         Returns:
#             (builder, checker, all_triplets, retrieval_scores)
#         """
#         from llm_triplet_extraction import LLMTripletExtractor
        
#         llm_extractor = LLMTripletExtractor(
#             api_key=self.api_key,
#             model=self.model_name,
#             temperature=0.0
#         )
        
#         if hasattr(self, 'api_base') and self.api_base:
#             llm_extractor.client.base_url = self.api_base
        
#         print(f"\n🤖 Extracting triplets from {min(self.max_docs, len(retrieved_docs))} docs...")
        
#         all_triplets = []
#         retrieval_scores = {}
        
#         # IMPROVEMENT: Extract from more docs
#         for i, doc in enumerate(retrieved_docs[:self.max_docs]):
#             if len(doc.strip()) < 20:
#                 continue
            
#             print(f"  Doc {i+1}/{min(self.max_docs, len(retrieved_docs))}: ", end='', flush=True)
            
#             # Extract more triplets per doc
#             triplets = llm_extractor.extract_triplets_single(
#                 text=doc,
#                 max_triplets=self.max_triplets_per_doc
#             )
            
#             # IMPROVEMENT: lower confidence threshold
#             triplets = [t for t in triplets if t.confidence > self.triplet_confidence_threshold]
            
#             print(f"extracted {len(triplets)} triplets")
            
#             all_triplets.extend(triplets)
            
#             # Store retrieval score (higher for earlier docs)
#             score = 1.0 - (i * 0.05)
#             for t in triplets:
#                 retrieval_scores[t.source_text] = score
        
#         print(f"✓ Total triplets before dedup: {len(all_triplets)}")
        
#         # IMPROVEMENT: Deduplicate triplets
#         deduplicated = TripletNormalizer.deduplicate_triplets(
#             all_triplets,
#             similarity_threshold=self.similarity_threshold
#         )
#         print(f"✓ Total triplets after dedup: {len(deduplicated)}")
        
#         # Build graph
#         builder = KnowledgeGraphBuilder(use_spacy=False)
#         builder.add_triplets(deduplicated)
        
#         print(f"✓ Graph built: {builder.graph.number_of_nodes()} nodes, {builder.graph.number_of_edges()} edges")
        
#         # IMPROVEMENT: Enhanced consistency checking
#         checker = ConsistencyChecker(builder.graph)
#         checker.check_comprehensive_mutual_exclusivity()
#         checker.check_comprehensive_transitivity()
#         checker.check_temporal_consistency()
        
#         return builder, checker, deduplicated, retrieval_scores
    
#     def _rank_and_select_triplets(
#         self,
#         triplets: List[Triplet],
#         query: str,
#         retrieval_scores: Dict[str, float]
#     ) -> List[Dict]:
#         """
#         Rank triplets by relevance and select top-K
        
#         IMPROVEMENT: Ranked injection instead of arbitrary 10
#         """
#         ranked = TripletRanker.rank_triplets(
#             triplets=triplets,
#             query=query,
#             retrieval_scores=retrieval_scores,
#             top_k=self.max_context_triplets
#         )
        
#         ranked_dicts = [
#             {
#                 "subject": t.subject,
#                 "relation": t.relation,
#                 "object": t.object,
#                 "confidence": t.confidence,
#                 "relevance_score": score
#             }
#             for t, score in ranked
#         ]
        
#         return ranked_dicts
    
#     def _graph_to_context(
#         self,
#         graph: nx.DiGraph,
#         ranked_triplets: List[Dict]
#     ) -> str:
#         """
#         Convert ranked triplets to natural language context
        
#         IMPROVEMENT: Line-by-line format for better LLM grounding
#         """
#         context_lines = []
        
#         # Use ranked triplets for better relevance
#         for triplet in ranked_triplets:
#             line = f"• {triplet['subject']} {triplet['relation']} {triplet['object']} (confidence: {triplet['confidence']:.2f})"
#             context_lines.append(line)
        
#         # Format as bullet list for better readability
#         context = "Validated Knowledge Facts:\n" + "\n".join(context_lines)
        
#         return context
    
#     def _calculate_enhanced_consistency_score(
#         self,
#         total_triplets: int,
#         contradictions: int,
#         inferences: int,
#         confidence_values: List[float]
#     ) -> float:
#         """
#         Enhanced consistency score calculation
        
#         Formula: (avg_confidence × 0.5) + ((1 - contradiction_ratio) × 0.3) + (inference_bonus × 0.2)
#         """
#         if total_triplets == 0:
#             return 0.0
        
#         # Average confidence of triplets
#         avg_confidence = np.mean(confidence_values) if confidence_values else 0.5
        
#         # Contradiction penalty
#         contradiction_ratio = contradictions / max(total_triplets, 1)
#         non_contradiction_score = 1.0 - contradiction_ratio
        
#         # Inference bonus
#         inference_ratio = inferences / max(total_triplets * 5, 1)
#         inference_bonus = min(1.0, inference_ratio)
        
#         # Weighted combination
#         score = (avg_confidence * 0.5) + (non_contradiction_score * 0.3) + (inference_bonus * 0.2)
        
#         return min(1.0, max(0.0, score))
    
#     def _calculate_factual_grounding_score(
#         self,
#         ranked_triplets: List[Dict],
#         answer: str
#     ) -> float:
#         """
#         Calculate how well answer is grounded in graph facts
        
#         IMPROVEMENT: New metric for factual grounding
#         """
#         if not ranked_triplets or not answer:
#             return 0.0
        
#         # Extract entities from answer
#         answer_lower = answer.lower()
        
#         # Count matches with triplet entities
#         matches = 0
#         for triplet in ranked_triplets:
#             subject = triplet['subject'].lower()
#             triplet_obj = triplet['object'].lower()
            
#             if subject in answer_lower or triplet_obj in answer_lower:
#                 matches += 1
        
#         grounding_score = matches / len(ranked_triplets) if ranked_triplets else 0.0
        
#         return min(1.0, grounding_score)
    
#     def query(self, question: str) -> GraphAugmentedResponse:
#         """
#         Query the Enhanced Graph-Augmented RAG system
#         """
#         print(f"\n{'='*80}")
#         print(f"QUERY: {question}")
#         print(f"{'='*80}")
        
#         # Step 1: Retrieve documents
#         base_response = self.base_rag.query(question)
#         retrieved_docs = base_response.retrieved_docs
        
#         print(f"📚 Retrieved {len(retrieved_docs)} documents")
        
#         # Step 2: Build graph with improved coverage
#         builder, checker, deduplicated_triplets, retrieval_scores = self._build_graph_from_retrieved(
#             retrieved_docs, question
#         )
        
#         print(f"🕸️  Graph: {builder.graph.number_of_nodes()} nodes, {builder.graph.number_of_edges()} edges")
#         print(f"⚠️  Contradictions: {len(checker.contradictions)}, Inferences: {len(checker.inferences)}")
        
#         # Step 3: Rank triplets for injection
#         ranked_triplets = self._rank_and_select_triplets(
#             deduplicated_triplets, question, retrieval_scores
#         )
        
#         print(f"📊 Selected {len(ranked_triplets)} ranked triplets for context")
        
#         # Step 4: Calculate enhanced consistency score
#         confidence_values = [t.confidence for t in deduplicated_triplets]
#         consistency_score = self._calculate_enhanced_consistency_score(
#             total_triplets=len(deduplicated_triplets),
#             contradictions=len(checker.contradictions),
#             inferences=len(checker.inferences),
#             confidence_values=confidence_values
#         )
        
#         print(f"✅ Consistency score: {consistency_score:.2f}")
        
#         # Step 5: Generate answer with ranked context
#         graph_context = self._graph_to_context(builder.graph, ranked_triplets)
        
#         if consistency_score >= self.consistency_threshold:
#             enhanced_question = f"""{graph_context}

# Question: {question}

# Provide a factual answer based on the validated knowledge above. Be concise and accurate."""
            
#             final_response = self.base_rag.query(enhanced_question)
#             answer = final_response.answer
#         else:
#             answer = f"[Warning: Low consistency ({consistency_score:.2f})] {base_response.answer}"
        
#         # Step 6: Calculate factual grounding
#         factual_grounding = self._calculate_factual_grounding_score(ranked_triplets, answer)
        
#         print(f"📌 Factual grounding: {factual_grounding:.2f}")
        
#         # Step 7: Create comprehensive response
#         response = GraphAugmentedResponse(
#             query=question,
#             answer=answer,
#             retrieved_docs=retrieved_docs,
#             graph_triplets=[
#                 {
#                     "subject": t.subject,
#                     "relation": t.relation,
#                     "object": t.object
#                 }
#                 for t in deduplicated_triplets[:20]
#             ],
#             ranked_triplets=ranked_triplets,
#             contradictions=checker.contradictions,
#             inferences=[
#                 {"subject": inf[0], "relation": inf[1], "object": inf[2]}
#                 for inf in checker.inferences
#             ],
#             consistency_score=consistency_score,
#             factual_grounding_score=factual_grounding,
#             metadata={
#                 "num_nodes": builder.graph.number_of_nodes(),
#                 "num_edges": builder.graph.number_of_edges(),
#                 "num_triplets_total": len(deduplicated_triplets),
#                 "num_triplets_ranked": len(ranked_triplets),
#                 "num_contradictions": len(checker.contradictions),
#                 "num_inferences": len(checker.inferences),
#                 "model": self.model_name
#             }
#         )
        
#         return response
    
#     def debug_triplets(self, question: str, top_k: int = 20):
#         """
#         Debug helper: Print top triplets for a query
        
#         IMPROVEMENT: Better debugging capabilities
#         """
#         base_response = self.base_rag.query(question)
#         builder, checker, triplets, retrieval_scores = self._build_graph_from_retrieved(
#             base_response.retrieved_docs, question
#         )
        
#         ranked = TripletRanker.rank_triplets(triplets, question, retrieval_scores, top_k=top_k)
        
#         print(f"\n📋 Top {len(ranked)} Triplets for Query: '{question}'")
#         print("-" * 80)
        
#         for i, (triplet, score) in enumerate(ranked, 1):
#             print(f"{i:2d}. [{score:.3f}] {triplet.subject} --{triplet.relation}--> {triplet.object}")
#             print(f"    Confidence: {triplet.confidence:.2f}")
        
#         print("-" * 80)


# # Example usage
# if __name__ == "__main__":
#     documents = [
#         "Albert Einstein was born in Germany in 1879. He was a theoretical physicist.",
#         "Einstein developed the theory of relativity in 1905. This theory revolutionized physics.",
#         "In 1921, Einstein received the Nobel Prize in Physics for his work on the photoelectric effect.",
#         "Einstein moved to the United States in 1933 and became a professor at Princeton University.",
#         "Einstein died in 1955 at the age of 76 in Princeton, New Jersey.",
#         "The theory of relativity consists of special relativity and general relativity.",
#         "Special relativity introduced the famous equation E=mc².",
#         "General relativity explains gravity as curved spacetime.",
#         "Einstein's work revolutionized our understanding of space and time.",
#         "Princeton University is located in New Jersey."
#     ]
    
#     API_KEY = "your-openrouter-api-key"
    
#     ga_rag = GraphAugmentedRAG(
#         api_key=API_KEY,
#         model_name="deepseek/deepseek-chat",
#         api_base="https://openrouter.ai/api/v1",
#         max_docs=10,
#         max_triplets_per_doc=15,
#         max_context_triplets=25,
#         triplet_confidence_threshold=0.5,
#     )
    
#     ga_rag.load_documents(documents)
    
#     test_queries = [
#         "When and where was Einstein born?",
#         "What did Einstein develop?",
#         "What is the theory of relativity?",
#         "What is E=mc²?"
#     ]
    
#     print("\n" + "="*80)
#     print("ENHANCED GRAPH-AUGMENTED RAG - EVALUATION")
#     print("="*80)
    
#     for query in test_queries:
#         response = ga_rag.query(query)
        
#         print(f"\n🟢 ANSWER: {response.answer}")
#         print(f"📊 SCORES: Consistency={response.consistency_score:.2f}, Grounding={response.factual_grounding_score:.2f}")
#         print(f"📈 METRICS: {response.metadata['num_triplets_ranked']} ranked triplets used")
        
#         # Debug triplets
#         ga_rag.debug_triplets(query, top_k=10)


"""
Enhanced Graph-Augmented RAG System - Improved Version
Addresses triplet coverage, graph context ranking, consistency scoring, and more
"""

import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import networkx as nx
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
import time

# Import from previous modules
from baseline_rag import BaselineRAG, RAGResponse
from graph_reasoning import KnowledgeGraphBuilder, LogicalConsistencyChecker, Triplet
from llm_triplet_extraction import LLMTripletExtractor, HybridTripletExtractor
from visualization_utils import RAGVisualizer

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "with", "about", "into",
    "there", "their", "this", "that", "those", "these", "from", "over",
    "after", "before", "because", "while", "where", "when", "have",
    "has", "had", "been", "being", "will", "would", "should", "could",
    "may", "might", "can", "cannot", "on", "in", "of", "to", "as",
    "by", "it", "its", "is", "are", "was", "were", "at", "not"
}


@dataclass
class GraphAugmentedResponse:
    """Enhanced response with graph reasoning information"""
    query: str
    answer: str
    retrieved_docs: List[str]
    graph_triplets: List[Dict]
    ranked_triplets: List[Dict]  # NEW: ranked by relevance
    contradictions: List[Dict]
    inferences: List[Dict]
    consistency_score: float
    factual_grounding_score: float  # NEW: how well-grounded in graph
    metadata: Dict[str, Any]


class TripletNormalizer:
    """Deduplicates and normalizes triplets"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        return text.lower().strip().replace("  ", " ")
    
    @staticmethod
    def text_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0.0 to 1.0)"""
        norm1 = TripletNormalizer.normalize_text(text1)
        norm2 = TripletNormalizer.normalize_text(text2)
        
        if norm1 == norm2:
            return 1.0
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    @staticmethod
    def deduplicate_triplets(triplets: List[Triplet], similarity_threshold: float = 0.85) -> List[Triplet]:
        """
        Remove duplicate triplets using similarity matching
        Keeps the one with highest confidence
        """
        if not triplets:
            return []
        
        # Sort by confidence descending
        sorted_triplets = sorted(triplets, key=lambda t: t.confidence, reverse=True)
        
        deduplicated = []
        seen_normalized = []
        
        for triplet in sorted_triplets:
            norm_key = (
                TripletNormalizer.normalize_text(triplet.subject),
                TripletNormalizer.normalize_text(triplet.relation),
                TripletNormalizer.normalize_text(triplet.object)
            )
            
            is_duplicate = False
            for seen_key in seen_normalized:
                # Check each component
                subject_sim = TripletNormalizer.text_similarity(norm_key[0], seen_key[0])
                relation_sim = TripletNormalizer.text_similarity(norm_key[1], seen_key[1])
                object_sim = TripletNormalizer.text_similarity(norm_key[2], seen_key[2])
                
                avg_sim = (subject_sim + relation_sim + object_sim) / 3
                
                if avg_sim >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(triplet)
                seen_normalized.append(norm_key)
        
        return deduplicated


class TripletRanker:
    """Ranks triplets by relevance to query"""
    
    @staticmethod
    def calculate_relevance_score(
        triplet: Triplet,
        query: str,
        retrieval_score: float = 1.0,
        mention_count: int = 1
    ) -> float:
        """
        Calculate combined relevance score
        Formula: confidence × (1 + mention_count*0.1) × retrieval_score × query_relevance
        """
        # Query relevance: how much query overlaps with triplet
        query_lower = query.lower()
        triplet_text = f"{triplet.subject} {triplet.relation} {triplet.object}".lower()
        
        query_words = set(query_lower.split())
        triplet_words = set(triplet_text.split())
        
        overlap = len(query_words & triplet_words)
        query_relevance = 1.0 + (overlap * 0.2)
        
        # Combine scores
        mention_factor = 1.0 + (mention_count * 0.1)
        
        score = (
            triplet.confidence * 
            mention_factor * 
            retrieval_score * 
            query_relevance
        )
        
        return score
    
    @staticmethod
    def rank_triplets(
        triplets: List[Triplet],
        query: str,
        retrieval_scores: Dict[str, float] = None,
        top_k: int = None
    ) -> List[Tuple[Triplet, float]]:
        """
        Rank triplets by relevance
        
        Returns:
            List of (triplet, score) tuples sorted by score descending
        """
        if retrieval_scores is None:
            retrieval_scores = defaultdict(lambda: 1.0)
        
        # Count mentions (triplets with same subject/object)
        mention_counts = defaultdict(int)
        for t in triplets:
            key = (t.subject, t.relation, t.object)
            mention_counts[key] += 1
        
        # Score each triplet
        scored = []
        for triplet in triplets:
            key = (triplet.subject, triplet.relation, triplet.object)
            mention_count = mention_counts[key]
            
            # Get retrieval score (default 1.0 if not provided)
            retrieval_score = retrieval_scores.get(triplet.source, 1.0)
            
            score = TripletRanker.calculate_relevance_score(
                triplet=triplet,
                query=query,
                retrieval_score=retrieval_score,
                mention_count=mention_count
            )
            
            scored.append((triplet, score))
        
        # Sort by score descending
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked


class EnhancedConsistencyChecker:
    """Enhanced consistency checking with more comprehensive rules"""
    
    # Expanded mutual exclusivity rules
    MUTUAL_EXCLUSIVITY_RULES = {
        "born": ["died", "passed_away", "was_assassinated"],
        "is_alive": ["died", "passed_away", "was_assassinated"],
        "created": ["destroyed", "demolished"],
        "is_author_of": ["is_translator_of"],  # Same person usually doesn't both author and translate
        "located_in": ["located_not_in"],
        "member_of": ["not_member_of"],
        "is_before": ["is_after"],
        "is_contemporary_with": ["is_before", "is_after"],
    }
    
    # Transitivity rules
    TRANSITIVITY_RULES = [
        ("is_part_of", "is_part_of"),
        ("located_in", "located_in"),
        ("member_of", "member_of"),
        ("is_ancestor_of", "is_ancestor_of"),
        ("is_supervisor_of", "is_supervisor_of"),
    ]
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.contradictions = []
        self.inferences = []
        self.confidence_scores = {}
    
    def check_comprehensive_mutual_exclusivity(self):
        """Check mutual exclusivity with logging"""
        violations = []
        
        for source_relation, conflicting_relations in self.MUTUAL_EXCLUSIVITY_RULES.items():
            # Find nodes with source_relation edges
            for u, v, data in self.graph.edges(data=True):
                if data.get("relation") == source_relation:
                    # Check if any conflicting edge exists
                    for conflict_rel in conflicting_relations:
                        if self.graph.has_edge(u, v) and self.graph[u][v].get("relation") == conflict_rel:
                            violations.append({
                                "type": "mutual_exclusivity",
                                "subject": u,
                                "object": v,
                                "conflicting_relations": [source_relation, conflict_rel]
                            })
        
        self.contradictions.extend(violations)
        return violations
    
    def check_comprehensive_transitivity(self):
        """Check transitivity with inference generation"""
        new_inferences = []
        
        for rel1, rel2 in self.TRANSITIVITY_RULES:
            # Find chains: A --rel1--> B --rel2--> C
            for u in self.graph.nodes():
                for v in self.graph.neighbors(u):
                    if self.graph[u][v].get("relation") == rel1:
                        for w in self.graph.neighbors(v):
                            if self.graph[v][w].get("relation") == rel2:
                                # Check if direct edge exists
                                if not self.graph.has_edge(u, w):
                                    new_inferences.append((u, rel1, w))
        
        self.inferences.extend(new_inferences)
        return new_inferences
    
    def check_temporal_consistency(self):
        """Detect temporal violations"""
        violations = []
        
        # Find temporal edges
        temporal_edges = []
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "")
            if any(t in rel.lower() for t in ["before", "after", "during", "year"]):
                temporal_edges.append((u, v, rel))
        
        # Simple check: if A is_before B and B is_before C, then A should be_before C
        # (simplified - would need actual temporal values in production)
        
        return violations


class GraphAugmentedRAG:
    """
    Enhanced Graph-Augmented RAG System with improved coverage and consistency
    
    Key Improvements:
    1. Increased triplet extraction (5-10 docs, 10-15 triplets per doc)
    2. Triplet deduplication using similarity matching
    3. Ranked triplet injection based on query relevance
    4. Enhanced consistency checking with expanded rules
    5. Factual grounding score
    6. Better debug output
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",  # Changed to OpenAI model
        # api_base: str = "https://openrouter.ai/api/v1",  # REMOVED: Not needed for OpenAI
        use_spacy: bool = False,
        consistency_threshold: float = 0.7,
        max_docs: int = 6,  # INCREASED: extract from more docs for better coverage
        max_triplets_per_doc: int = 4,  # INCREASED: more triplets per doc
        max_context_triplets: int = 4,  # INCREASED: inject more facts
        triplet_confidence_threshold: float = 0.6,  # ADJUSTED: slightly higher for quality
        similarity_threshold: float = 0.85,  # For deduplication
        use_hybrid_triplets: bool = True,
        caching_enabled: bool = True,
        min_context_triplets_for_graph: int = 5,
        grounding_warning_threshold: float = 0.2,
        entity_expansion_k: int = 2,
        visualize_graphs: bool = False,
        graph_viz_output_dir: str = "results/graphs",
        graph_viz_max_nodes: int = 7,
    ):
        """Initialize Enhanced Graph-Augmented RAG with OpenAI API"""
        
        self.api_key = api_key
        # self.api_base = api_base  # REMOVED for OpenAI
        self.model_name = model_name
        self.use_spacy = use_spacy
        self.consistency_threshold = consistency_threshold
        self.max_docs = max_docs
        self.max_triplets_per_doc = max_triplets_per_doc
        self.max_context_triplets = max_context_triplets
        self.triplet_confidence_threshold = triplet_confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_triplets = use_hybrid_triplets
        self.caching_enabled = caching_enabled
        self.min_context_triplets_for_graph = min_context_triplets_for_graph
        self.grounding_warning_threshold = grounding_warning_threshold
        self.entity_expansion_k = entity_expansion_k
        self.visualize_graphs = visualize_graphs
        self.graph_viz_output_dir = Path(graph_viz_output_dir)
        self.graph_viz_max_nodes = graph_viz_max_nodes
        self.triplet_cache: Dict[str, List[Triplet]] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.triplet_extractor = LLMTripletExtractor(
            api_key=api_key,
            model=model_name,
            temperature=0.4
        )
        self._hybrid_extractor = (
            HybridTripletExtractor(api_key=api_key, use_spacy=use_spacy)
            if use_hybrid_triplets else None
        )
        self.graph_visualizer = RAGVisualizer()
        if self.visualize_graphs:
            self.graph_viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        # os.environ["OPENAI_API_BASE"] = api_base  # REMOVED for OpenAI
        
        from baseline_rag import BaselineRAG
        self.base_rag = BaselineRAG(
            api_key=api_key,
            model_name=model_name
            # api_base=api_base  # REMOVED for OpenAI
        )
        
        # ===== COMMENTED OUT: Old OpenRouter implementation =====
        # self.api_base = api_base
        # os.environ["OPENAI_API_BASE"] = api_base
        # self.base_rag = BaselineRAG(
        #     api_key=api_key,
        #     model_name=model_name,
        #     api_base=api_base
        # )
        
        print("✓ Enhanced Graph-Augmented RAG initialized")
        print(f"  - Max docs per query: {max_docs}")
        print(f"  - Max triplets per doc: {max_triplets_per_doc}")
        print(f"  - Max context triplets: {max_context_triplets}")
        print(f"  - Hybrid extraction: {'on' if use_hybrid_triplets else 'off'}")
        print(f"  - Triplet caching: {'on' if caching_enabled else 'off'}")
        print(f"  - Min facts for graph answer: {min_context_triplets_for_graph}")
        print(f"  - Grounding warning threshold: {grounding_warning_threshold}")
        print(f"  - Entity expansion per probe: {entity_expansion_k}")
        print(
            f"  - Graph visualization: {'on' if visualize_graphs else 'off'}"
            f" (dir: {self.graph_viz_output_dir})"
        )
    
    def load_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Load documents into the system"""
        self.base_rag.load_documents(documents, metadatas)
        self.base_rag.build_qa_chain()

    # ------------------------------------------------------------------
    # Query understanding helpers
    # ------------------------------------------------------------------
    def _tokenize_for_matching(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
        return [tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS]

    def _extract_query_terms(self, question: str) -> Set[str]:
        return set(self._tokenize_for_matching(question))

    def _expand_query_terms_with_graph(
        self,
        base_terms: Set[str],
        graph: nx.DiGraph,
        max_expansions: int = 8
    ) -> Set[str]:
        """Expand query terms using neighboring graph nodes (graph as expert)."""
        if not base_terms:
            return set()

        expanded = set(base_terms)
        for node in list(graph.nodes())[: self.graph_viz_max_nodes]:
            node_tokens = set(self._tokenize_for_matching(str(node)))
            if node_tokens & base_terms:
                expanded |= node_tokens
                if len(expanded) >= max_expansions:
                    break
            # bring in immediate neighbors when node matches
            if node_tokens & base_terms:
                for neighbor in graph.neighbors(node):
                    expanded |= set(self._tokenize_for_matching(str(neighbor)))
        return expanded or base_terms

    def _filter_triplets_by_query(
        self,
        ranked_triplets: List[Dict],
        expanded_query_terms: Set[str]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Remove triplets that do not intersect with the query/graph vocabulary."""
        if not ranked_triplets or not expanded_query_terms:
            return ranked_triplets, {
                "kept": len(ranked_triplets),
                "dropped": 0,
                "precision": 1.0 if ranked_triplets else 0.0,
                "fallback_used": False,
                "query_terms": sorted(expanded_query_terms)
            }

        kept: List[Dict] = []
        dropped: List[Dict] = []

        for triplet in ranked_triplets:
            subject_tokens = set(self._tokenize_for_matching(triplet["subject"]))
            object_tokens = set(self._tokenize_for_matching(triplet["object"]))
            rel_tokens = set(self._tokenize_for_matching(triplet.get("relation", "")))

            if subject_tokens & expanded_query_terms or object_tokens & expanded_query_terms:
                kept.append(triplet)
            elif rel_tokens & expanded_query_terms:
                kept.append(triplet)
            else:
                dropped.append(triplet)

        fallback_used = False
        if not kept:
            kept = ranked_triplets
            dropped = []
            fallback_used = True

        precision = len(kept) / max(1, len(kept) + len(dropped))

        filter_stats = {
            "kept": len(kept),
            "dropped": len(dropped),
            "precision": precision,
            "fallback_used": fallback_used,
            "query_terms": sorted(expanded_query_terms)
        }

        return kept, filter_stats

    def _compute_graph_coverage_metrics(
        self,
        query_terms: Set[str],
        graph: nx.DiGraph,
        ranked_triplets: List[Dict]
    ) -> Dict[str, float]:
        if not query_terms:
            return {
                "node_coverage": 1.0,
                "fact_coverage": 1.0,
                "triplet_density": float(graph.number_of_edges()) / max(1, graph.number_of_nodes()),
            }

        matched_node_terms: Set[str] = set()
        for node in graph.nodes():
            node_tokens = set(self._tokenize_for_matching(str(node)))
            matched_node_terms |= (node_tokens & query_terms)

        matched_fact_terms: Set[str] = set()
        for triplet in ranked_triplets:
            matched_fact_terms |= set(self._tokenize_for_matching(triplet["subject"])) & query_terms
            matched_fact_terms |= set(self._tokenize_for_matching(triplet["object"])) & query_terms

        node_coverage = len(matched_node_terms) / len(query_terms)
        fact_coverage = len(matched_fact_terms) / len(query_terms)
        triplet_density = float(graph.number_of_edges()) / max(1, graph.number_of_nodes())

        return {
            "node_coverage": min(1.0, node_coverage),
            "fact_coverage": min(1.0, fact_coverage),
            "triplet_density": triplet_density,
        }

    def _hash_document(self, text: str) -> str:
        """Create a stable hash for caching triplets."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _extract_triplets_for_doc(self, doc_text: str, query: str) -> List[Triplet]:
        """Extract triplets from a single document with hybrid + coverage boosts."""
        if len(doc_text.strip()) < 20:
            return []

        try:
            primary_triplets = self.triplet_extractor.extract_triplets_single(
                text=doc_text,
                max_triplets=self.max_triplets_per_doc
            )
        except Exception as exc:
            print(f"⚠️ Triplet extraction failed: {exc}")
            return []

        # Optional spaCy augmentation for extra coverage
        if self.use_hybrid_triplets and self._hybrid_extractor:
            try:
                spacy_triplets = self._hybrid_extractor.extract_with_spacy(doc_text)
            except AttributeError:
                spacy_triplets = self._hybrid_extractor.extract(doc_text)
            except Exception:
                spacy_triplets = []
            primary_triplets.extend(spacy_triplets)

        # If coverage is low, fall back to query-focused extraction
        min_coverage = max(6, self.max_triplets_per_doc // 2)
        if query and len(primary_triplets) < min_coverage:
            try:
                contextual_triplets = self.triplet_extractor.extract_with_context(doc_text, query)
                primary_triplets.extend(contextual_triplets)
            except Exception:
                pass

        filtered = [
            t for t in primary_triplets
            if getattr(t, "confidence", 1.0) >= self.triplet_confidence_threshold
        ]

        deduped = TripletNormalizer.deduplicate_triplets(
            filtered,
            similarity_threshold=self.similarity_threshold
        )

        return deduped[: self.max_triplets_per_doc]

    def _get_triplets_with_cache(self, doc_text: str, query: str) -> Tuple[List[Triplet], bool]:
        """Return triplets for doc, using cache when enabled."""
        if not self.caching_enabled:
            return self._extract_triplets_for_doc(doc_text, query), False

        cache_key = self._hash_document(doc_text)
        if cache_key in self.triplet_cache:
            self.cache_stats["hits"] += 1
            return list(self.triplet_cache[cache_key]), True

        triplets = self._extract_triplets_for_doc(doc_text, query)
        self.triplet_cache[cache_key] = list(triplets)
        self.cache_stats["misses"] += 1
        return triplets, False

    def _visualize_graph(self, graph: nx.DiGraph, question: str) -> str:
        """Render and save the knowledge graph if visualization is enabled."""
        if not self.visualize_graphs or graph.number_of_nodes() == 0:
            return ""

        self.graph_viz_output_dir.mkdir(parents=True, exist_ok=True)
        safe_question = re.sub(r"[^a-zA-Z0-9]+", "_", question.lower()).strip("_") or "query"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.graph_viz_output_dir / f"{safe_question[:60]}_{timestamp}.png"

        try:
            self.graph_visualizer.plot_knowledge_graph(
                graph,
                title=f"Knowledge Graph – {question[:60]}",
                max_nodes=self.graph_viz_max_nodes,
                save_path=str(file_path),
                show=False,
            )
            return str(file_path)
        except Exception as exc:
            print(f"⚠️ Failed to visualize graph: {exc}")
            return ""

    def _extract_candidate_entities(
        self,
        question: str,
        docs: List[str],
        max_entities: int = 4
    ) -> List[str]:
        """Derive entity cues for retrieval expansion using simple heuristics."""
        text_window = "\n".join([question] + docs[:2])
        capitalized = re.findall(r"[A-Z][\w]+(?:\s+[A-Z][\w]+)?", text_window)
        candidates = Counter([c.strip() for c in capitalized if len(c.strip()) > 2])

        if not candidates:
            tokens = [t for t in re.split(r"[^a-zA-Z0-9]", text_window) if len(t) > 4]
            candidates = Counter(tokens)

        return [item for item, _ in candidates.most_common(max_entities)]

    def _augment_retrieved_docs(
        self,
        question: str,
        docs: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Expand retrieval set with KG-aware entity probes when coverage is low."""
        unique_docs = []
        seen = set()
        for doc in docs:
            normalized = doc.strip()
            if normalized and normalized not in seen:
                unique_docs.append(normalized)
                seen.add(normalized)

        if len(unique_docs) >= self.max_docs or not hasattr(self.base_rag, "retrieve_similar_documents"):
            return unique_docs[: self.max_docs], {"extra_docs": 0, "entities_used": []}

        total_tokens = sum(len(doc.split()) for doc in unique_docs)
        token_threshold = self.max_docs * 120
        if total_tokens >= token_threshold:
            return unique_docs[: self.max_docs], {"extra_docs": 0, "entities_used": []}

        candidate_entities = self._extract_candidate_entities(
            question,
            unique_docs,
            max_entities=max(2, self.entity_expansion_k * 2)
        )

        extra_docs = 0
        used_entities = []
        for entity in candidate_entities:
            similar = self.base_rag.retrieve_similar_documents(entity, k=self.entity_expansion_k)
            if not similar:
                continue
            for doc in similar:
                normalized = doc.strip()
                if not normalized or normalized in seen:
                    continue
                unique_docs.append(normalized)
                seen.add(normalized)
                extra_docs += 1
                used_entities.append(entity)
                if len(unique_docs) >= self.max_docs:
                    break
            if len(unique_docs) >= self.max_docs:
                break

        return unique_docs[: self.max_docs], {
            "extra_docs": extra_docs,
            "entities_used": used_entities,
            "candidate_entities": candidate_entities,
        }

    def _filter_retrieved_docs_by_query(
        self,
        question: str,
        docs: List[str],
        min_overlap_tokens: int = 2,
        min_ratio: float = 0.15
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Heuristic filter to keep only retrieved docs that overlap the query.

        Keeps docs that have at least `min_overlap_tokens` overlapping token matches
        with the query, or whose overlap ratio (over query tokens) >= `min_ratio`.

        Falls back to returning the original docs (limited to `max_docs`) when
        no documents meet the threshold.
        """
        query_terms = self._extract_query_terms(question)
        if not docs or not query_terms:
            return docs[: self.max_docs], {"kept": len(docs[: self.max_docs]), "dropped": 0, "fallback": False}

        kept = []
        dropped = []

        for doc in docs:
            doc_tokens = set(self._tokenize_for_matching(doc))
            overlap = len(doc_tokens & query_terms)
            ratio = overlap / max(1, len(query_terms))

            if overlap >= min_overlap_tokens or ratio >= min_ratio:
                kept.append(doc)
            else:
                dropped.append(doc)

        fallback = False
        if not kept:
            # Fallback: keep the top-k retrieved docs to avoid empty retrieval
            kept = docs[: self.max_docs]
            dropped = []
            fallback = True

        # Ensure we respect max_docs
        kept = kept[: self.max_docs]

        stats = {
            "kept": len(kept),
            "dropped": len(dropped),
            "fallback": fallback,
            "min_overlap_tokens": min_overlap_tokens,
            "min_ratio": min_ratio,
        }

        return kept, stats
    
    def _build_graph_from_retrieved(self, retrieved_docs: List[str], query: str) -> Tuple:
        """Build a knowledge graph with caching and hybrid coverage.

        Returns:
            (builder, checker, triplets, retrieval_scores, coverage_stats)
        """
        target_docs = min(self.max_docs, len(retrieved_docs))
        print(f"\n🤖 Extracting triplets from {target_docs} docs (cache enabled: {self.caching_enabled})...")

        all_triplets: List[Triplet] = []
        retrieval_scores: Dict[str, float] = {}
        doc_triplet_counts: List[int] = []
        cache_hits = 0

        for i, doc in enumerate(retrieved_docs[: self.max_docs]):
            triplets, cache_hit = self._get_triplets_with_cache(doc, query)
            cache_hits += int(cache_hit)
            doc_triplet_counts.append(len(triplets))

            status = "cache" if cache_hit else "fresh"
            print(
                f"  Doc {i+1}/{target_docs} ({status}): {len(triplets)} triplets",
                flush=True,
            )

            if not triplets:
                continue

            all_triplets.extend(triplets)

            score = max(0.2, 1.0 - (i * 0.05))
            for triplet in triplets:
                if not getattr(triplet, "source", ""):
                    try:
                        setattr(triplet, "source", f"doc_{i}")
                    except Exception:
                        pass
                source_key = getattr(triplet, "source", f"doc_{i}")
                retrieval_scores[source_key] = max(retrieval_scores.get(source_key, 0.0), score)

        print(f"✓ Total triplets before dedup: {len(all_triplets)}")

        deduplicated = TripletNormalizer.deduplicate_triplets(
            all_triplets,
            similarity_threshold=self.similarity_threshold
        )
        print(f"✓ Total triplets after dedup: {len(deduplicated)}")

        inferred_triplets = []
        if deduplicated and len(deduplicated) < self.max_context_triplets:
            inferred_triplets = self.triplet_extractor.infer_missing_triplets(
                deduplicated,
                max_inferences=3
            )
            if inferred_triplets:
                print(f"  ↳ Added {len(inferred_triplets)} inferred triplets for coverage")
                deduplicated = TripletNormalizer.deduplicate_triplets(
                    deduplicated + inferred_triplets,
                    similarity_threshold=self.similarity_threshold
                )

        builder = KnowledgeGraphBuilder(use_spacy=False)
        builder.add_triplets(deduplicated)

        print(f"✓ Graph built: {builder.graph.number_of_nodes()} nodes, {builder.graph.number_of_edges()} edges")

        checker = EnhancedConsistencyChecker(builder.graph)
        checker.check_comprehensive_mutual_exclusivity()
        checker.check_comprehensive_transitivity()
        checker.check_temporal_consistency()

        coverage_stats = {
            "cache_hits": cache_hits,
            "docs_processed": len(doc_triplet_counts),
            "avg_triplets_per_doc": float(np.mean(doc_triplet_counts)) if doc_triplet_counts else 0.0,
            "inferred_triplets": len(inferred_triplets),
        }

        return builder, checker, deduplicated, retrieval_scores, coverage_stats
    
    def _rank_and_select_triplets(
        self,
        triplets: List[Triplet],
        query: str,
        retrieval_scores: Dict[str, float]
    ) -> List[Dict]:
        """
        Rank triplets by relevance and select top-K
        
        IMPROVEMENT: Ranked injection instead of arbitrary 10
        """
        ranked = TripletRanker.rank_triplets(
            triplets=triplets,
            query=query,
            retrieval_scores=retrieval_scores,
            top_k=self.max_context_triplets
        )
        
        ranked_dicts = [
            {
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "confidence": t.confidence,
                "relevance_score": score,
                "source": getattr(t, "source", "")
            }
            for t, score in ranked
        ]
        
        return ranked_dicts
    
    def _graph_to_context(
        self,
        graph: nx.DiGraph,
        ranked_triplets: List[Dict]
    ) -> str:
        """
        Convert ranked triplets to natural language context
        
        IMPROVEMENT: Line-by-line format for better LLM grounding
        """
        context_lines = []
        
        # Use ranked triplets for better relevance
        for idx, triplet in enumerate(ranked_triplets, start=1):
            line = (
                f"Fact {idx}: {triplet['subject']} {triplet['relation']} {triplet['object']} "
                f"(confidence: {triplet['confidence']:.2f})"
            )
            context_lines.append(line)
        
        # Format as bullet list for better readability
        context = "Validated Knowledge Facts:\n" + "\n".join(context_lines)
        
        return context
    
    def _calculate_enhanced_consistency_score(
        self,
        total_triplets: int,
        contradictions: int,
        inferences: int,
        confidence_values: List[float]
    ) -> float:
        """
        Enhanced consistency score calculation
        
        Formula: (avg_confidence × 0.5) + ((1 - contradiction_ratio) × 0.3) + (inference_bonus × 0.2)
        """
        if total_triplets == 0:
            return 0.0
        
        # Average confidence of triplets
        avg_confidence = np.mean(confidence_values) if confidence_values else 0.5
        
        # Contradiction penalty
        contradiction_ratio = contradictions / max(total_triplets, 1)
        non_contradiction_score = 1.0 - contradiction_ratio
        
        # Inference bonus
        inference_ratio = inferences / max(total_triplets * 5, 1)
        inference_bonus = min(1.0, inference_ratio)
        
        # Weighted combination
        score = (avg_confidence * 0.5) + (non_contradiction_score * 0.3) + (inference_bonus * 0.2)
        
        return min(1.0, max(0.0, score))
    
    def _calculate_factual_grounding_score(
        self,
        ranked_triplets: List[Dict],
        answer: str
    ) -> float:
        """
        Calculate how well answer is grounded in graph facts
        
        IMPROVEMENT: New metric for factual grounding
        """
        if not ranked_triplets or not answer:
            return 0.0
        
        # Extract entities from answer
        answer_lower = answer.lower()
        
        # Count matches with triplet entities
        matches = 0
        for triplet in ranked_triplets:
            subject = triplet['subject'].lower()
            triplet_obj = triplet['object'].lower()
            
            if subject in answer_lower or triplet_obj in answer_lower:
                matches += 1
        
        grounding_score = matches / len(ranked_triplets) if ranked_triplets else 0.0
        
        return min(1.0, grounding_score)
    
    def query(self, question: str) -> GraphAugmentedResponse:
        """
        Query the Enhanced Graph-Augmented RAG system
        """
        print(f"\n{'='*80}")
        print(f"QUERY: {question}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve documents
        base_response = self.base_rag.query(question)
        initial_docs = base_response.retrieved_docs or []
        retrieved_docs, retrieval_aug = self._augment_retrieved_docs(question, initial_docs)
        
        print(f"📚 Retrieved {len(retrieved_docs)} documents (initial {len(initial_docs)})")
        if retrieval_aug.get("extra_docs"):
            print(
                f"🔁 Retrieval expansion: +{retrieval_aug['extra_docs']} docs via "
                f"{len(set(retrieval_aug.get('entities_used', [])))} entity probes"
            )
        
        # Step 2: Filter retrieved docs by query relevance, then build graph
        filtered_docs, per_query_filter = self._filter_retrieved_docs_by_query(
            question, retrieved_docs, min_overlap_tokens=2, min_ratio=0.15
        )

        if per_query_filter.get("dropped"):
            print(f"🔎 Per-query retrieval filter kept {per_query_filter['kept']} / {per_query_filter['kept'] + per_query_filter['dropped']} docs (fallback={per_query_filter['fallback']})")

        (
            builder,
            checker,
            deduplicated_triplets,
            retrieval_scores,
            coverage_stats,
        ) = self._build_graph_from_retrieved(filtered_docs, question)
        
        print(f"🕸️  Graph: {builder.graph.number_of_nodes()} nodes, {builder.graph.number_of_edges()} edges")
        print(f"⚠️  Contradictions: {len(checker.contradictions)}, Inferences: {len(checker.inferences)}")
        print(
            f"📈 Coverage: {coverage_stats['avg_triplets_per_doc']:.2f} triplets/doc, "
            f"cache hits: {coverage_stats['cache_hits']}"
        )
        graph_viz_path = self._visualize_graph(builder.graph, question)
        if graph_viz_path:
            print(f"🗺️  Graph visualization saved: {graph_viz_path}")
        
        # Step 3: Rank triplets for injection
        ranked_triplets = self._rank_and_select_triplets(
            deduplicated_triplets, question, retrieval_scores
        )
        query_terms = self._extract_query_terms(question)
        expanded_terms = self._expand_query_terms_with_graph(query_terms, builder.graph)
        ranked_triplets, filter_stats = self._filter_triplets_by_query(
            ranked_triplets, expanded_terms
        )
        coverage_metrics = self._compute_graph_coverage_metrics(
            query_terms, builder.graph, ranked_triplets
        )
        
        print(f"📊 Selected {len(ranked_triplets)} ranked triplets for context")
        if filter_stats.get("dropped"):
            print(
                f"🔎 Query filter kept {filter_stats['kept']} / {filter_stats['kept'] + filter_stats['dropped']} "
                f"triplets (precision {filter_stats['precision']:.2f})"
            )
        print(
            f"📐 Coverage: facts {coverage_metrics['fact_coverage']:.2f}, nodes {coverage_metrics['node_coverage']:.2f}"
        )
        if len(ranked_triplets) < self.min_context_triplets_for_graph:
            print(
                f"⚠️ Coverage below threshold ({len(ranked_triplets)} < {self.min_context_triplets_for_graph});"
                " will fall back to baseline if needed"
            )
        
        # Step 4: Calculate enhanced consistency score
        confidence_values = [t.confidence for t in deduplicated_triplets]
        consistency_score = self._calculate_enhanced_consistency_score(
            total_triplets=len(deduplicated_triplets),
            contradictions=len(checker.contradictions),
            inferences=len(checker.inferences),
            confidence_values=confidence_values
        )
        
        print(f"✅ Consistency score: {consistency_score:.2f}")
        
        # Step 5: Generate answer with ranked context
        graph_context = self._graph_to_context(builder.graph, ranked_triplets)
        
        coverage_ok = len(ranked_triplets) >= self.min_context_triplets_for_graph
        use_graph_answer = consistency_score >= self.consistency_threshold and coverage_ok
        answer_source = "graph" if use_graph_answer else "baseline"

        if use_graph_answer:
            enhanced_question = f"""{graph_context}

You must use ONLY the validated facts above.
Question: {question}

Instructions:
- Reference the supporting fact numbers in parentheses (e.g., Fact 2) for every claim.
- If the facts are insufficient, reply verbatim: "I cannot confirm based on the validated facts."
- Do not introduce information that is not explicitly in the facts.

Provide a concise answer:
"""
            
            final_response = self.base_rag.query(enhanced_question)
            answer = final_response.answer
        else:
            if not coverage_ok:
                answer = f"[Graph coverage low ({len(ranked_triplets)} facts)] {base_response.answer}"
            else:
                answer = f"[Warning: Low consistency ({consistency_score:.2f})] {base_response.answer}"
            answer_source = "baseline"
        
        # Step 6: Calculate factual grounding
        factual_grounding = self._calculate_factual_grounding_score(ranked_triplets, answer)
        grounding_backoff = False
        if (
            answer_source == "graph"
            and factual_grounding < self.grounding_warning_threshold
        ):
            answer = (
                f"[Low grounding {factual_grounding:.2f}] {base_response.answer}"
            )
            factual_grounding = 0.0
            answer_source = "baseline"
            grounding_backoff = True
        
        print(f"📌 Factual grounding: {factual_grounding:.2f}")
        
        # Step 7: Create comprehensive response
        response = GraphAugmentedResponse(
            query=question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            graph_triplets=[
                {
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object
                }
                for t in deduplicated_triplets[:20]
            ],
            ranked_triplets=ranked_triplets,
            contradictions=checker.contradictions,
            inferences=[
                {"subject": inf[0], "relation": inf[1], "object": inf[2]}
                for inf in checker.inferences
            ],
            consistency_score=consistency_score,
            factual_grounding_score=factual_grounding,
            metadata={
                "num_nodes": builder.graph.number_of_nodes(),
                "num_edges": builder.graph.number_of_edges(),
                "num_triplets_total": len(deduplicated_triplets),
                "num_triplets_ranked": len(ranked_triplets),
                "num_contradictions": len(checker.contradictions),
                "num_inferences": len(checker.inferences),
                "cache_hits": coverage_stats["cache_hits"],
                "avg_triplets_per_doc": coverage_stats["avg_triplets_per_doc"],
                "inferred_triplets": coverage_stats["inferred_triplets"],
                "coverage_ok": coverage_ok,
                "answer_source": answer_source,
                "grounding_backoff": grounding_backoff,
                "retrieval_expansion": retrieval_aug,
                "graph_viz_path": graph_viz_path,
                "model": self.model_name,
                "context_filter": filter_stats,
                "context_filter_precision": filter_stats.get("precision", 0.0),
                "question_term_coverage": coverage_metrics.get("fact_coverage", 0.0),
                "node_term_coverage": coverage_metrics.get("node_coverage", 0.0),
                "triplet_density": coverage_metrics.get("triplet_density", 0.0),
                "query_terms_used": filter_stats.get("query_terms", [])
            }
        )
        
        return response
    
    def debug_triplets(self, question: str, top_k: int = 20):
        """
        Debug helper: Print top triplets for a query
        
        IMPROVEMENT: Better debugging capabilities
        """
        base_response = self.base_rag.query(question)
        builder, checker, triplets, retrieval_scores, _ = self._build_graph_from_retrieved(
            base_response.retrieved_docs, question
        )
        
        ranked = TripletRanker.rank_triplets(triplets, question, retrieval_scores, top_k=top_k)
        
        print(f"\n📋 Top {len(ranked)} Triplets for Query: '{question}'")
        print("-" * 80)
        
        for i, (triplet, score) in enumerate(ranked, 1):
            print(f"{i:2d}. [{score:.3f}] {triplet.subject} --{triplet.relation}--> {triplet.object}")
            print(f"    Confidence: {triplet.confidence:.2f}")
        
        print("-" * 80)


# Example usage
if __name__ == "__main__":
    documents = [
        "Albert Einstein was born in Germany in 1879. He was a theoretical physicist.",
        "Einstein developed the theory of relativity in 1905. This theory revolutionized physics.",
        "In 1921, Einstein received the Nobel Prize in Physics for his work on the photoelectric effect.",
        "Einstein moved to the United States in 1933 and became a professor at Princeton University.",
        "Einstein died in 1955 at the age of 76 in Princeton, New Jersey.",
        "The theory of relativity consists of special relativity and general relativity.",
        "Special relativity introduced the famous equation E=mc².",
        "General relativity explains gravity as curved spacetime.",
        "Einstein's work revolutionized our understanding of space and time.",
        "Princeton University is located in New Jersey."
    ]
    
    API_KEY = "your-openrouter-api-key"
    
    ga_rag = GraphAugmentedRAG(
        api_key=API_KEY,
        model_name="deepseek/deepseek-chat",
        api_base="https://openrouter.ai/api/v1",
        max_docs=3,
        max_triplets_per_doc=6,
        max_context_triplets=10,
        triplet_confidence_threshold=0.5,
    )
    
    ga_rag.load_documents(documents)
    
    test_queries = [
        "When and where was Einstein born?",
        "What did Einstein develop?",
        "What is the theory of relativity?",
        "What is E=mc²?"
    ]
    
    print("\n" + "="*80)
    print("ENHANCED GRAPH-AUGMENTED RAG - EVALUATION")
    print("="*80)
    
    for query in test_queries:
        response = ga_rag.query(query)
        
        print(f"\n🟢 ANSWER: {response.answer}")
        print(f"📊 SCORES: Consistency={response.consistency_score:.2f}, Grounding={response.factual_grounding_score:.2f}")
        print(f"📈 METRICS: {response.metadata['num_triplets_ranked']} ranked triplets used")
        
        # Debug triplets
        ga_rag.debug_triplets(query, top_k=10)