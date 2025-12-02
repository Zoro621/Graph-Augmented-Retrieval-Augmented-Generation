"""
LLM-Based Knowledge Triplet Extraction
More accurate than rule-based extraction, uses GPT to extract structured facts
"""

import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import openai
from openai import OpenAI


# @dataclass
# class Triplet:
#     """Represents a knowledge triplet"""
#     subject: str
#     relation: str
#     object: str
#     confidence: float = 1.0
#     source_text: str = ""

from dataclasses import dataclass

@dataclass
class Triplet:
    """Represents a knowledge triplet"""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str = ""  # Changed from source_text for consistency
    
    def __hash__(self):
        """Make Triplet hashable for set operations"""
        return hash((self.subject.lower(), self.relation.lower(), self.object.lower()))
    
    def __eq__(self, other):
        """Check equality (case-insensitive)"""
        if not isinstance(other, Triplet):
            return False
        return (
            self.subject.lower() == other.subject.lower() and
            self.relation.lower() == other.relation.lower() and
            self.object.lower() == other.object.lower()
        )


class LLMTripletExtractor:
    """
    Uses LLM to extract structured knowledge triplets from text
    More accurate and flexible than rule-based extraction
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        """
        Initialize LLM-based extractor
        
        Args:
            api_key: OpenAI API key
            model: Model to use for extraction
            temperature: Sampling temperature (0 for deterministic)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
    
    def extract_triplets_single(
        self,
        text: str,
        max_triplets: int = 10
    ) -> List[Triplet]:
        """
        Extract triplets from a single text passage
        
        Args:
            text: Input text
            max_triplets: Maximum number of triplets to extract
            
        Returns:
            List of Triplet objects
        """
        prompt = f"""Extract factual knowledge triplets from the following text. 
Each triplet should be in the form: (Subject, Relation, Object)

Rules:
1. Extract only factual, verifiable relationships explicitly stated in the text
2. Subjects and Objects MUST be specific named entities (people, places, organizations) or concrete concepts
3. Relations should be normalized verbs or predicates (e.g., 'born_in', 'works_for', 'created', 'located_in', 'has_profession')
4. Avoid pronouns - always use full entity names
5. Normalize entities (e.g., "Einstein" and "Albert Einstein" should be consistent)
6. Extract up to {max_triplets} most important triplets

Text: "{text}"

Return ONLY a JSON array of triplets in this format:
[
  {{"subject": "Entity1", "relation": "normalized_verb", "object": "Entity2", "confidence": 0.95}},
  ...
]

Include confidence (0.0-1.0) based on how explicitly the fact is stated.
Do not include any explanation, only the JSON array."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured knowledge from text. You always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            # Clean up response (remove markdown code blocks if present)
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            # Parse JSON
            triplets_data = json.loads(content)
            
            # Convert to Triplet objects
            triplets = []
            for t in triplets_data:
                triplet = Triplet(
                    subject=t.get("subject", "").strip(),
                    relation=t.get("relation", "").strip(),
                    object=t.get("object", "").strip(),
                    confidence=t.get("confidence", 1.0),
                    source=text[:100]
                )
                triplets.append(triplet)
            
            return triplets
            
        except Exception as e:
            print(f"Error extracting triplets: {e}")
            return []
    
    def extract_triplets_batch(
        self,
        texts: List[str],
        max_triplets_per_text: int = 10
    ) -> List[Triplet]:
        """
        Extract triplets from multiple texts
        
        Args:
            texts: List of text passages
            max_triplets_per_text: Max triplets per passage
            
        Returns:
            Combined list of all triplets
        """
        all_triplets = []
        
        for i, text in enumerate(texts):
            print(f"Processing text {i+1}/{len(texts)}...", end='\r')
            triplets = self.extract_triplets_single(text, max_triplets_per_text)
            all_triplets.extend(triplets)
        
        print(f"\n✓ Extracted {len(all_triplets)} triplets from {len(texts)} texts")
        
        return all_triplets
    
    def extract_with_context(
        self,
        text: str,
        query: str = None
    ) -> List[Triplet]:
        """
        Extract triplets relevant to a specific query
        
        Args:
            text: Input text
            query: Query to focus extraction on
            
        Returns:
            List of query-relevant triplets
        """
        if query:
            context_prompt = f"\nFocus on facts relevant to this question: {query}"
        else:
            context_prompt = ""
        
        prompt = f"""Extract factual knowledge triplets from the following text.
Each triplet should be in the form: (Subject, Relation, Object){context_prompt}

Text: "{text}"

Return ONLY a JSON array of triplets in this format:
[
  {{"subject": "...", "relation": "...", "object": "...", "confidence": 0.95}},
  ...
]

Include a confidence score (0.0-1.0) for each triplet based on how directly it's stated in the text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured knowledge. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            triplets_data = json.loads(content)
            
            triplets = []
            for t in triplets_data:
                triplet = Triplet(
                    subject=t.get("subject", "").strip(),
                    relation=t.get("relation", "").strip(),
                    object=t.get("object", "").strip(),
                    confidence=t.get("confidence", 1.0),
                    source=text[:100]
                )
                triplets.append(triplet)
            
            return triplets
            
        except Exception as e:
            print(f"Error extracting triplets: {e}")
            return []
    
    def verify_triplet_consistency(
        self,
        triplet1: Triplet,
        triplet2: Triplet
    ) -> Dict[str, Any]:
        """
        Use LLM to check if two triplets are consistent
        
        Args:
            triplet1: First triplet
            triplet2: Second triplet
            
        Returns:
            Dict with consistency analysis
        """
        prompt = f"""Analyze if these two facts are logically consistent:

Fact 1: {triplet1.subject} {triplet1.relation} {triplet1.object}
Fact 2: {triplet2.subject} {triplet2.relation} {triplet2.object}

Return ONLY a JSON object with this format:
{{
  "consistent": true/false,
  "reason": "brief explanation",
  "confidence": 0.0-1.0
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at logical reasoning. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            result = json.loads(content)
            return result
            
        except Exception as e:
            print(f"Error verifying consistency: {e}")
            return {"consistent": True, "reason": "error", "confidence": 0.5}
    
    def infer_missing_triplets(
        self,
        existing_triplets: List[Triplet],
        max_inferences: int = 5
    ) -> List[Triplet]:
        """
        Use LLM to infer logical consequences from existing triplets
        
        Args:
            existing_triplets: List of known triplets
            max_inferences: Maximum number of inferences to make
            
        Returns:
            List of inferred triplets
        """
        # Format existing triplets
        facts = "\n".join([
            f"- {t.subject} {t.relation} {t.object}"
            for t in existing_triplets[:20]  # Limit context
        ])
        
        prompt = f"""Given these known facts:

{facts}

What logical inferences can you make? Extract up to {max_inferences} new facts that logically follow.

Return ONLY a JSON array of inferred triplets:
[
  {{"subject": "...", "relation": "...", "object": "...", "confidence": 0.0-1.0}},
  ...
]

Only include inferences with high confidence."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at logical reasoning and inference. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            inferences_data = json.loads(content)
            
            inferences = []
            for inf in inferences_data:
                triplet = Triplet(
                    subject=inf.get("subject", "").strip(),
                    relation=inf.get("relation", "").strip(),
                    object=inf.get("object", "").strip(),
                    confidence=inf.get("confidence", 0.8),
                    source="[INFERRED]"
                )
                inferences.append(triplet)
            
            print(f"✓ Made {len(inferences)} logical inferences")
            
            return inferences
            
        except Exception as e:
            print(f"Error making inferences: {e}")
            return []


class HybridTripletExtractor:
    """
    Combines spaCy rule-based extraction with LLM-based extraction
    Falls back to LLM when spaCy extraction yields few results
    """
    
    def __init__(
        self,
        api_key: str,
        use_spacy: bool = True,
        spacy_threshold: int = 3
    ):
        """
        Initialize hybrid extractor
        
        Args:
            api_key: OpenAI API key
            use_spacy: Whether to try spaCy first
            spacy_threshold: Minimum spaCy triplets before using LLM
        """
        self.llm_extractor = LLMTripletExtractor(api_key)
        self.use_spacy = use_spacy
        self.spacy_threshold = spacy_threshold
        
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ Hybrid extractor initialized (spaCy + LLM)")
            except:
                self.nlp = None
                print("⚠ spaCy not available, using LLM only")
        else:
            self.nlp = None
            print("✓ Hybrid extractor initialized (LLM only)")
    
    def extract_with_spacy(self, text: str) -> List[Triplet]:
        """Simple spaCy extraction"""
        if not self.nlp:
            return []
        
        triplets = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    subjects = [child for child in token.children 
                               if child.dep_ in ("nsubj", "nsubjpass")]
                    objects = [child for child in token.children 
                              if child.dep_ in ("dobj", "pobj", "attr")]
                    
                    for subj in subjects:
                        for obj in objects:
                            triplet = Triplet(
                                subject=subj.text.strip(),
                                relation=token.lemma_.strip(),
                                object=obj.text.strip(),
                                confidence=0.7,
                                source=text[:100]
                            )
                            triplets.append(triplet)
        
        return triplets
    
    def extract(self, text: str) -> List[Triplet]:
        """
        Extract triplets using hybrid approach
        
        Args:
            text: Input text
            
        Returns:
            List of triplets
        """
        # Try spaCy first if enabled
        if self.use_spacy and self.nlp:
            spacy_triplets = self.extract_with_spacy(text)
            
            if len(spacy_triplets) >= self.spacy_threshold:
                print(f"✓ Extracted {len(spacy_triplets)} triplets with spaCy")
                return spacy_triplets
        
        # Fall back to LLM
        print("→ Using LLM extraction...")
        llm_triplets = self.llm_extractor.extract_triplets_single(text)
        
        return llm_triplets


# Example usage
if __name__ == "__main__":
    # Sample text
    sample_text = """
    Albert Einstein was born in Germany in 1879. He developed the theory of relativity 
    in 1905. Einstein received the Nobel Prize in Physics in 1921 for his work on the 
    photoelectric effect. He moved to the United States in 1933 and worked at Princeton 
    University until his death in 1955.
    """
    
    # Initialize extractor
    API_KEY = "your-openai-api-key-here"
    
    print("="*60)
    print("LLM-BASED TRIPLET EXTRACTION DEMO")
    print("="*60)
    
    extractor = LLMTripletExtractor(api_key=API_KEY)
    
    # Extract triplets
    print("\n1. Extracting triplets from text...")
    triplets = extractor.extract_triplets_single(sample_text)
    
    print(f"\n✓ Extracted {len(triplets)} triplets:")
    for i, t in enumerate(triplets, 1):
        print(f"   {i}. ({t.subject}) --[{t.relation}]--> ({t.object})")
    
    # Extract with query context
    print("\n2. Extracting with query context...")
    query = "When did Einstein receive the Nobel Prize?"
    context_triplets = extractor.extract_with_context(sample_text, query)
    
    print(f"\n✓ Extracted {len(context_triplets)} relevant triplets:")
    for i, t in enumerate(context_triplets, 1):
        print(f"   {i}. ({t.subject}) --[{t.relation}]--> ({t.object}) [conf: {t.confidence}]")
    
    # Make inferences
    if len(triplets) > 0:
        print("\n3. Making logical inferences...")
        inferences = extractor.infer_missing_triplets(triplets, max_inferences=3)
        
        print(f"\n✓ Made {len(inferences)} inferences:")
        for i, t in enumerate(inferences, 1):
            print(f"   {i}. ({t.subject}) --[{t.relation}]--> ({t.object}) [conf: {t.confidence}]")
    
    print("\n" + "="*60)
    print("Demo complete!")