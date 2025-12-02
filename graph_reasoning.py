# """
# Knowledge Graph Construction and Reasoning Module
# Week 11: Graph-Augmented RAG Implementation
# """

# import networkx as nx
# from typing import List, Dict, Tuple, Set, Any
# from dataclasses import dataclass
# import spacy
# import re
# from collections import defaultdict


# @dataclass
# class Triplet:
#     """Represents a knowledge triplet (subject, relation, object)"""
#     subject: str
#     relation: str
#     object: str
#     source: str = ""  # Source document for traceability
#     confidence: float = 1.0
    
#     def __hash__(self):
#         return hash((self.subject, self.relation, self.object))
    
#     def __eq__(self, other):
#         return (self.subject, self.relation, self.object) == \
#                (other.subject, other.relation, other.object)


# class KnowledgeGraphBuilder:
#     """
#     Builds and manages knowledge graphs from retrieved text
#     """
    
#     def __init__(self, use_spacy: bool = True):
#         """
#         Initialize graph builder
        
#         Args:
#             use_spacy: Whether to use spaCy for NER and dependency parsing
#         """
#         self.graph = nx.DiGraph()
#         self.triplets: Set[Triplet] = set()
        
#         # Load spaCy model if available
#         self.nlp = None
#         if use_spacy:
#             try:
#                 self.nlp = spacy.load("en_core_web_sm")
#                 print("✓ SpaCy model loaded")
#             except:
#                 print("⚠ SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
    
#     def extract_triplets_simple(self, text: str) -> List[Triplet]:
#         """
#         Simple rule-based triplet extraction
#         Extracts triplets using basic NLP patterns
#         """
#         triplets = []
        
#         if not self.nlp:
#             return triplets
        
#         doc = self.nlp(text)
        
#         # Extract triplets from dependency parse
#         for sent in doc.sents:
#             # Find subject-verb-object patterns
#             for token in sent:
#                 if token.pos_ == "VERB":
#                     # Find subject
#                     subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
#                     # Find objects
#                     objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                    
#                     for subj in subjects:
#                         for obj in objects:
#                             triplet = Triplet(
#                                 subject=subj.text,
#                                 relation=token.lemma_,
#                                 object=obj.text,
#                                 source=text[:100]
#                             )
#                             triplets.append(triplet)
        
#         return triplets
    
#     def extract_triplets_with_entities(self, text: str) -> List[Triplet]:
#         """
#         Extract triplets focusing on named entities
#         Better for factual knowledge extraction
#         """
#         triplets = []
        
#         if not self.nlp:
#             return triplets
        
#         doc = self.nlp(text)
        
#         # Extract named entities
#         entities = [(ent.text, ent.label_) for ent in doc.ents]
        
#         # Create entity-based triplets
#         for i, (ent1, label1) in enumerate(entities):
#             for j, (ent2, label2) in enumerate(entities):
#                 if i != j:
#                     # Find connecting verbs between entities
#                     for token in doc:
#                         if token.pos_ == "VERB":
#                             # Simple proximity check
#                             if ent1 in token.sent.text and ent2 in token.sent.text:
#                                 triplet = Triplet(
#                                     subject=ent1,
#                                     relation=f"{token.lemma_}_{label1}_to_{label2}",
#                                     object=ent2,
#                                     source=text[:100]
#                                 )
#                                 triplets.append(triplet)
        
#         return triplets
    
#     def add_triplets(self, triplets: List[Triplet]):
#         """Add triplets to the knowledge graph"""
#         for triplet in triplets:
#             self.triplets.add(triplet)
            
#             # Add nodes and edges to graph
#             self.graph.add_node(triplet.subject, type="entity")
#             self.graph.add_node(triplet.object, type="entity")
#             self.graph.add_edge(
#                 triplet.subject,
#                 triplet.object,
#                 relation=triplet.relation,
#                 confidence=triplet.confidence
#             )
    
#     def build_from_documents(self, documents: List[str]) -> int:
#         """
#         Build knowledge graph from list of documents
        
#         Returns:
#             Number of triplets extracted
#         """
#         all_triplets = []
        
#         for doc in documents:
#             # Try both extraction methods
#             triplets1 = self.extract_triplets_simple(doc)
#             triplets2 = self.extract_triplets_with_entities(doc)
            
#             all_triplets.extend(triplets1)
#             all_triplets.extend(triplets2)
        
#         self.add_triplets(all_triplets)
        
#         print(f"✓ Extracted {len(self.triplets)} unique triplets")
#         print(f"✓ Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
#         return len(self.triplets)


# class LogicalConsistencyChecker:
#     """
#     Applies logical reasoning and consistency checks on knowledge graphs
#     """
    
#     def __init__(self, graph: nx.DiGraph):
#         self.graph = graph
#         self.contradictions = []
#         self.inferences = []
    
#     def check_mutual_exclusivity(self, rules: Dict[str, List[str]]) -> List[Dict]:
#         """
#         Check for mutually exclusive relations
        
#         Args:
#             rules: Dict mapping relations to mutually exclusive relations
#                    e.g., {"is_alive": ["is_dead", "died_in"]}
#         """
#         contradictions = []
        
#         for node in self.graph.nodes():
#             # Get all outgoing edges
#             edges = self.graph.out_edges(node, data=True)
#             relations = [edge[2].get("relation", "") for edge in edges]
            
#             # Check for contradictions
#             for rel, exclusive_rels in rules.items():
#                 if rel in relations:
#                     for excl_rel in exclusive_rels:
#                         if excl_rel in relations:
#                             contradictions.append({
#                                 "type": "mutual_exclusivity",
#                                 "node": node,
#                                 "relation1": rel,
#                                 "relation2": excl_rel
#                             })
        
#         self.contradictions.extend(contradictions)
#         return contradictions
    
#     def check_transitivity(self, transitive_relations: List[str]) -> List[Tuple]:
#         """
#         Infer new facts using transitivity
        
#         Args:
#             transitive_relations: List of relations that are transitive
#                                  e.g., ["is_part_of", "is_subclass_of"]
#         """
#         inferences = []
        
#         for relation in transitive_relations:
#             # Find all edges with this relation
#             edges = [
#                 (u, v) for u, v, data in self.graph.edges(data=True)
#                 if data.get("relation") == relation
#             ]
            
#             # Check for transitive paths
#             for e1_start, e1_end in edges:
#                 for e2_start, e2_end in edges:
#                     if e1_end == e2_start:
#                         # Found transitive relationship
#                         if not self.graph.has_edge(e1_start, e2_end):
#                             inferences.append((e1_start, relation, e2_end))
        
#         self.inferences.extend(inferences)
#         return inferences
    
#     def check_temporal_consistency(self) -> List[Dict]:
#         """
#         Check for temporal inconsistencies
#         (e.g., birth year > death year, event order violations)
#         """
#         temporal_contradictions = []
        
#         # Extract temporal relations
#         temporal_edges = [
#             (u, v, data) for u, v, data in self.graph.edges(data=True)
#             if any(temp in data.get("relation", "").lower() 
#                    for temp in ["born", "died", "before", "after"])
#         ]
        
#         # Simple year extraction and comparison
#         year_pattern = r'\b(19|20)\d{2}\b'
        
#         for u, v, data in temporal_edges:
#             years_in_u = re.findall(year_pattern, str(u))
#             years_in_v = re.findall(year_pattern, str(v))
            
#             if years_in_u and years_in_v:
#                 year_u = int(years_in_u[0])
#                 year_v = int(years_in_v[0])
                
#                 # Check for impossible orderings
#                 if "born" in data["relation"] and "died" in data["relation"]:
#                     if year_u > year_v:
#                         temporal_contradictions.append({
#                             "type": "temporal_order",
#                             "issue": f"Birth year {year_u} > death year {year_v}",
#                             "nodes": (u, v)
#                         })
        
#         self.contradictions.extend(temporal_contradictions)
#         return temporal_contradictions
    
#     def get_validated_subgraph(self) -> nx.DiGraph:
#         """
#         Return subgraph with contradictory edges removed
#         """
#         validated_graph = self.graph.copy()
        
#         # Remove edges involved in contradictions
#         for contradiction in self.contradictions:
#             if contradiction["type"] == "mutual_exclusivity":
#                 node = contradiction["node"]
#                 # Remove edges with conflicting relations
#                 edges_to_remove = [
#                     (u, v) for u, v, data in validated_graph.out_edges(node, data=True)
#                     if data.get("relation") in [contradiction["relation1"], contradiction["relation2"]]
#                 ]
#                 validated_graph.remove_edges_from(edges_to_remove)
        
#         return validated_graph
    
#     def get_consistency_report(self) -> Dict[str, Any]:
#         """Generate comprehensive consistency report"""
#         return {
#             "total_nodes": self.graph.number_of_nodes(),
#             "total_edges": self.graph.number_of_edges(),
#             "contradictions_found": len(self.contradictions),
#             "inferences_made": len(self.inferences),
#             "contradiction_details": self.contradictions,
#             "inference_details": self.inferences
#         }


# # Example usage
# if __name__ == "__main__":
#     # Sample documents
#     sample_docs = [
#         "Einstein was born in Germany in 1879.",
#         "Einstein developed the theory of relativity.",
#         "Einstein died in 1955 in the United States.",
#         "The theory of relativity revolutionized physics."
#     ]
    
#     # Build knowledge graph
#     builder = KnowledgeGraphBuilder(use_spacy=True)
#     builder.build_from_documents(sample_docs)
    
#     # Run consistency checks
#     checker = LogicalConsistencyChecker(builder.graph)
    
#     # Check for contradictions
#     contradictions = checker.check_mutual_exclusivity({
#         "born": ["died"],
#         "is_alive": ["died"]
#     })
    
#     # Infer new facts
#     inferences = checker.check_transitivity(["developed", "is_part_of"])
    
#     # Check temporal consistency
#     temporal_issues = checker.check_temporal_consistency()
    
#     # Get report
#     report = checker.get_consistency_report()
    
#     print("\n" + "="*60)
#     print("KNOWLEDGE GRAPH CONSISTENCY REPORT")
#     print("="*60)
#     print(f"Nodes: {report['total_nodes']}")
#     print(f"Edges: {report['total_edges']}")
#     print(f"Contradictions: {report['contradictions_found']}")
#     print(f"Inferences: {report['inferences_made']}")

"""
Patched Knowledge Graph Construction and Reasoning Module
- Normalizes entities/relations
- Uses NetworkX MultiDiGraph to preserve multiple relations
- Improved SVO + entity-based extraction
- Batch processing with spaCy pipe
- Optional Neo4j push (if neo4j driver installed)
- Confidence aggregation and provenance
- Robust contradiction, transitivity, and temporal checks
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Any, Optional, Iterable
from collections import defaultdict

import networkx as nx

try:
    import spacy
except Exception:
    spacy = None

# Optional Neo4j support
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except Exception:
    GraphDatabase = None
    NEO4J_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()


@dataclass(frozen=True)
class Triplet:
    subject: str = field(compare=True)
    relation: str = field(compare=True)
    object: str = field(compare=True)
    source: str = field(compare=False, default="")
    confidence: float = field(compare=False, default=1.0)

    def __post_init__(self):
        # Normalize fields and enforce immutability
        object.__setattr__(self, 'subject', normalize_text(self.subject))
        object.__setattr__(self, 'relation', normalize_text(self.relation))
        object.__setattr__(self, 'object', normalize_text(self.object))
        # Truncate source for storage
        object.__setattr__(self, 'source', (self.source or "")[:500])
        # Ensure confidence is float
        try:
            object.__setattr__(self, 'confidence', float(self.confidence))
        except Exception:
            object.__setattr__(self, 'confidence', 1.0)


class KnowledgeGraphBuilder:
    """Builds and manages a knowledge graph from text documents.

    By default this uses NetworkX.MultiDiGraph to preserve multiple relations
    between the same node pair. It optionally supports pushing the graph to
    Neo4j if the neo4j Python driver is installed and the user supplies
    connection credentials.
    """

    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        self.graph = nx.MultiDiGraph()
        self.triplets: Set[Triplet] = set()
        self._nlp = None
        self.neo4j_driver = None
        self.neo4j_enabled = False

        if use_spacy:
            if spacy is None:
                logger.warning("spaCy not installed. Install with: pip install spacy")
            else:
                try:
                    self._nlp = spacy.load(spacy_model)
                    logger.info("✓ SpaCy model loaded: %s", spacy_model)
                except Exception as e:
                    logger.warning("Could not load spaCy model '%s': %s", spacy_model, e)
                    self._nlp = None

    # --------------------- Extraction ---------------------
    def extract_triplets_simple(self, doc_text: str) -> List[Triplet]:
        """Rule-based SVO extraction using dependency parse and subtree search."""
        triplets: List[Triplet] = []
        if not self._nlp:
            return triplets

        doc = self._nlp(doc_text)
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    verb_lemma = token.lemma_.lower()

                    # Subjects: nsubj, nsubjpass (prefer left-side children)
                    subj_candidates = [t for t in token.lefts if t.dep_ in ("nsubj", "nsubjpass")]
                    if not subj_candidates:
                        subj_candidates = [t for t in sent if t.dep_ in ("nsubj", "nsubjpass")]

                    # Objects: dobj, pobj, attr; fallback to noun tokens in subtree
                    obj_candidates = [t for t in token.rights if t.dep_ in ("dobj", "pobj", "attr")]
                    if not obj_candidates:
                        obj_candidates = [t for t in token.subtree if t.pos_ in ("NOUN", "PROPN") and t != token]

                    for subj in subj_candidates:
                        for obj in obj_candidates:
                            s_text = subj.text
                            o_text = obj.text
                            r_text = verb_lemma
                            triplets.append(Triplet(subject=s_text, relation=r_text, object=o_text, source=doc_text[:500]))
        return triplets

    def extract_triplets_with_entities(self, doc_text: str) -> List[Triplet]:
        """Entity-focused extraction. Only links entities that appear within the same sentence.

        This is conservative (reduces false positives) and better for factual knowledge.
        """
        triplets: List[Triplet] = []
        if not self._nlp:
            return triplets

        doc = self._nlp(doc_text)
        entities = list(doc.ents)
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities):
                if i == j:
                    continue
                # Only if they are in the same sentence
                if ent1.sent != ent2.sent:
                    continue
                sent = ent1.sent
                # pick verbs in the sentence and use their lemma as relation
                verbs = [tok for tok in sent if tok.pos_ == "VERB"]
                for v in verbs:
                    relation = v.lemma_.lower()
                    triplets.append(Triplet(subject=ent1.text, relation=relation, object=ent2.text, source=doc_text[:500]))
        return triplets

    def extract_triplets_from_documents(self, documents: Iterable[str], batch: bool = True, batch_size: int = 16) -> List[Triplet]:
        """Extract triplets from iterable of documents; use spaCy.pipe for speed when available."""
        all_triplets: List[Triplet] = []
        if not self._nlp:
            return all_triplets

        if batch:
            for doc in self._nlp.pipe(documents, batch_size=batch_size):
                text = doc.text
                # Use the two extractors on the same spaCy doc to avoid re-parsing
                all_triplets.extend(self.extract_triplets_simple(text))
                all_triplets.extend(self.extract_triplets_with_entities(text))
        else:
            for text in documents:
                all_triplets.extend(self.extract_triplets_simple(text))
                all_triplets.extend(self.extract_triplets_with_entities(text))

        return all_triplets

    # --------------------- Graph Construction ---------------------
    def add_triplets(self, triplets: Iterable[Triplet], aggregate_confidence: bool = True):
        """Add triplets to the graph. For MultiDiGraph we preserve multiple edges.

        If aggregate_confidence is True and the same triplet (identical s,r,o) is added
        multiple times, we keep the maximum confidence and track counts in node attributes.
        """
        # Track mentions count for (s,r,o)
        occurrences: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(lambda: {"count": 0, "max_conf": 0.0, "sources": []})

        for t in triplets:
            occurrences[(t.subject, t.relation, t.object)]["count"] += 1
            occurrences[(t.subject, t.relation, t.object)]["max_conf"] = max(occurrences[(t.subject, t.relation, t.object)]["max_conf"], float(t.confidence))
            occurrences[(t.subject, t.relation, t.object)]["sources"].append(t.source)
            self.triplets.add(t)

        # Add to graph preserving multi-edges
        for (s, r, o), meta in occurrences.items():
            self.graph.add_node(s, type="entity")
            self.graph.add_node(o, type="entity")
            # Add an edge and store aggregated metadata
            self.graph.add_edge(s, o, relation=r, confidence=float(meta["max_conf"] or 1.0), mentions=int(meta["count"]), sources=meta["sources"])

        logger.info("✓ Added %d triplets to graph (unique: %d)", sum(v["count"] for v in occurrences.values()), len(occurrences))

    def build_from_documents(self, documents: List[str], top_k_per_query: Optional[int] = None) -> int:
        """Convenience wrapper: extract from documents and add to graph.

        If top_k_per_query is provided, documents are expected to already be sorted per query
        and only the top_k_per_query documents per query should be passed in. This parameter
        is kept for compatibility with RAG-style workflows.
        """
        triplets = self.extract_triplets_from_documents(documents)
        self.add_triplets(triplets)
        logger.info("✓ Graph has %d nodes and %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        return len(self.triplets)

    # --------------------- Neo4j integration (optional) ---------------------
    def enable_neo4j(self, uri: str, user: str, password: str):
        if not NEO4J_AVAILABLE:
            raise RuntimeError("neo4j driver not available. Install with: pip install neo4j")
        self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        self.neo4j_enabled = True
        logger.info("✓ Neo4j support enabled (uri=%s)", uri)

    def push_to_neo4j(self, batch_size: int = 1000):
        if not self.neo4j_enabled or self.neo4j_driver is None:
            raise RuntimeError("Neo4j not enabled. Call enable_neo4j() first.")

        with self.neo4j_driver.session() as session:
            # Create nodes and relationships in batches
            edges = list(self.graph.edges(data=True, keys=False))
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                def tx_fn(tx, batch):
                    for u, v, data in batch:
                        # Merge nodes by name and create relationship with properties
                        tx.run(
                            "MERGE (a:Entity {name:$u}) MERGE (b:Entity {name:$v}) "+
                            "MERGE (a)-[r:RELATED {relation:$rel}]->(b) "+
                            "ON CREATE SET r.confidence = $conf, r.mentions = $mentions, r.sources = $sources "+
                            "ON MATCH SET r.confidence = apoc.coll.max([coalesce(r.confidence,0), $conf]), r.mentions = r.mentions + $mentions",
                            u=u, v=v, rel=data.get('relation'), conf=data.get('confidence', 1.0), mentions=data.get('mentions', 1), sources=data.get('sources', [])
                        )
                session.write_transaction(tx_fn, batch)
        logger.info("✓ Pushed graph to Neo4j (edges=%d)", self.graph.number_of_edges())


class LogicalConsistencyChecker:
    """Performs logical consistency checks and inference on a graph (NetworkX MultiDiGraph supported)."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.contradictions: List[Dict[str, Any]] = []
        self.inferences: List[Tuple[str, str, str]] = []

    def check_mutual_exclusivity(self, rules: Dict[str, List[str]]) -> List[Dict]:
        """Check for nodes that have mutually exclusive relations.

        rules: mapping like {"is_alive": ["died", "is_dead"]}
        Uses word boundary matching to avoid false positives.
        """
        contradictions: List[Dict] = []
        for node in self.graph.nodes():
            # For MultiDiGraph, out_edges yields (u,v,data)
            out = list(self.graph.out_edges(node, data=True))
            relations = [d.get('relation', '') for *_, d in out]
            for rule_rel, exclusive_list in rules.items():
                for existing_rel in relations:
                    # Use word boundary matching to avoid false positives
                    if re.search(rf'\b{re.escape(rule_rel)}\b', existing_rel):
                        for excl in exclusive_list:
                            for existing_rel2 in relations:
                                if re.search(rf'\b{re.escape(excl)}\b', existing_rel2):
                                    contradictions.append({
                                        'type': 'mutual_exclusivity',
                                        'node': node,
                                        'relation1': existing_rel,
                                        'relation2': existing_rel2
                                    })
        self.contradictions.extend(contradictions)
        return contradictions

    def check_transitivity(self, transitive_relations: List[str], add_inferred: bool = True) -> List[Tuple]:
        """Infer transitive relationships for relations present in transitive_relations.

        Example: if A -is_part_of-> B and B -is_part_of-> C then infer A -is_part_of-> C
        """
        inferences: List[Tuple[str, str, str]] = []
        for trans_rel in transitive_relations:
            # find edges where relation matches trans_rel (word boundary)
            edges = [(u, v) for u, v, data in self.graph.edges(data=True) 
                    if re.search(rf'\b{re.escape(trans_rel)}\b', data.get('relation', ''))]
            # build adjacency
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
            for a in adj:
                for b in adj[a]:
                    for c in adj.get(b, []):
                        # Check if a->c edge with this exact relation already exists
                        exists = any((a == x and c == y and re.search(rf'\b{re.escape(trans_rel)}\b', data.get('relation', ''))) 
                                   for x, y, data in self.graph.edges(data=True))
                        if not exists:
                            inferences.append((a, trans_rel, c))
                            if add_inferred:
                                # add inferred edge with low confidence
                                self.graph.add_edge(a, c, relation=f"{trans_rel}_inferred", confidence=0.3, mentions=0, sources=["inference"])
        self.inferences.extend(inferences)
        return inferences

    def check_temporal_consistency(self) -> List[Dict]:
        """Check for simple temporal contradictions like birth year > death year.

        This looks for edges that contain 'born' and 'died' (substring match) for the same subject.
        It extracts 4-digit years with a non-capturing group to get full matches.
        """
        temporal_contradictions: List[Dict] = []
        year_pattern = r"\b(?:19|20)\d{2}\b"

        # Map subject -> list of (relation, target_text)
        subj_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            rel = data.get('relation', '')
            subj_map[u].append((rel, v))

        for subj, rels in subj_map.items():
            births = [t for r, t in rels if 'born' in r]
            deaths = [t for r, t in rels if 'died' in r or 'death' in r]
            # if we have both, try to parse years
            for b in births:
                for d in deaths:
                    years_b = re.findall(year_pattern, str(b)) or re.findall(year_pattern, str(subj))
                    years_d = re.findall(year_pattern, str(d)) or re.findall(year_pattern, str(subj))
                    if years_b and years_d:
                        try:
                            yb = int(years_b[0])
                            yd = int(years_d[0])
                            if yb > yd:
                                temporal_contradictions.append({
                                    'type': 'temporal_order',
                                    'issue': f'birth year {yb} > death year {yd}',
                                    'nodes': (subj, subj)
                                })
                        except Exception:
                            continue
        self.contradictions.extend(temporal_contradictions)
        return temporal_contradictions

    def get_validated_subgraph(self) -> nx.MultiDiGraph:
        """Return a copy of the graph with contradictory edges removed.

        For MultiDiGraph we remove specific keyed edges where relation substrings match
        the contradictory relations.
        """
        validated_graph = self.graph.copy()
        for contradiction in self.contradictions:
            if contradiction.get('type') == 'mutual_exclusivity':
                node = contradiction.get('node')
                r1 = contradiction.get('relation1')
                r2 = contradiction.get('relation2')
                # remove edges from node whose relation contains r1 or r2
                for u, v, key, data in list(validated_graph.out_edges(node, keys=True, data=True)):
                    if r1 in data.get('relation', '') or r2 in data.get('relation', ''):
                        validated_graph.remove_edge(u, v, key=key)
        return validated_graph

    def get_consistency_report(self) -> Dict[str, Any]:
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'contradictions_found': len(self.contradictions),
            'inferences_made': len(self.inferences),
            'contradiction_details': self.contradictions,
            'inference_details': self.inferences
        }


# --------------------- Example usage (if run directly) ---------------------
if __name__ == '__main__':
    sample_docs = [
        "Albert Einstein was born in Ulm, Germany in 1879.",
        "Einstein developed the theory of relativity.",
        "Einstein died in 1955 in the United States.",
        "The theory of relativity revolutionized physics."
    ]

    builder = KnowledgeGraphBuilder(use_spacy=(spacy is not None))
    if builder._nlp:
        builder.build_from_documents(sample_docs)
        checker = LogicalConsistencyChecker(builder.graph)
        contradictions = checker.check_mutual_exclusivity({
            'born': ['died', 'death'],
            'is_alive': ['died', 'is_dead']
        })
        inferences = checker.check_transitivity(['part_of', 'is_part_of', 'developed'])
        temporal_issues = checker.check_temporal_consistency()
        report = checker.get_consistency_report()
        print('\n' + '='*60)
        print('KNOWLEDGE GRAPH CONSISTENCY REPORT')
        print('='*60)
        print(f"Nodes: {report['total_nodes']}")
        print(f"Edges: {report['total_edges']}")
        print(f"Contradictions: {report['contradictions_found']}")
        print(f"Inferences: {report['inferences_made']}")
    else:
        logger.warning('spaCy not available; extraction skipped')
