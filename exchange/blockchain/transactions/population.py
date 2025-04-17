from typing import Dict, Any, Optional, List
import datetime
import json
from ..security import sign_data, verify_signature, generate_random_id

class PopulationIndicatorTransaction:
    """Transaction for adding population indicators for cohort analysis."""
    def __init__(self, cohort_id: str = None, markers: Dict[str, Any] = None, 
                submitter_id: str = None):
        self.transaction_type = "ADD_POPULATION_INDICATOR"
        self.indicator_id = generate_random_id("POP-")
        self.cohort_id = cohort_id or generate_random_id("COH-")
        self.markers = markers or {}  # e.g., {"age": ">65", "gender": "F", "ethnicity": "Asian"}
        self.submitter_id = submitter_id
        self.timestamp = datetime.datetime.now()
        self.signature = None
        self.additional_data = {}
        self.related_evidence = []  # Evidence IDs this indicator relates to
    
    def add_marker(self, key: str, value: Any) -> None:
        """Add a population marker."""
        self.markers[key] = value
    
    def add_data(self, key: str, value: Any) -> None:
        """Add additional data to the indicator."""
        self.additional_data[key] = value
    
    def relate_to_evidence(self, evidence_id: str) -> None:
        """Associate this population indicator with evidence."""
        if evidence_id not in self.related_evidence:
            self.related_evidence.append(evidence_id)
    
    def sign(self, private_key: str) -> None:
        """Sign the transaction with submitter's private key."""
        data_to_sign = json.dumps(self.to_dict(include_signature=False), sort_keys=True)
        self.signature = sign_data(data_to_sign, private_key)
    
    def verify(self, public_key: str) -> bool:
        """Verify the transaction signature."""
        if not self.signature:
            return False
            
        data_to_verify = json.dumps(self.to_dict(include_signature=False), sort_keys=True)
        return verify_signature(data_to_verify, self.signature, public_key)
    
    def to_dict(self, include_signature: bool = True) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "type": self.transaction_type,
            "indicator_id": self.indicator_id,
            "cohort_id": self.cohort_id,
            "markers": self.markers,
            "submitter_id": self.submitter_id,
            "timestamp": str(self.timestamp),
            "additional_data": self.additional_data,
            "related_evidence": self.related_evidence
        }
        
        if include_signature and self.signature:
            result["signature"] = self.signature
            
        return result
    
    @classmethod
    def find_similar_cohorts(cls, target_markers: Dict[str, Any], population_indicators: List['PopulationIndicatorTransaction'], 
                            similarity_threshold: float = 0.7) -> List[Dict]:
        """Find similar cohorts based on marker similarity."""
        results = []
        
        for indicator in population_indicators:
            similarity_score = cls._calculate_similarity(target_markers, indicator.markers)
            
            if similarity_score >= similarity_threshold:
                results.append({
                    "indicator_id": indicator.indicator_id,
                    "cohort_id": indicator.cohort_id,
                    "similarity_score": similarity_score,
                    "markers": indicator.markers
                })
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results
    
    @staticmethod
    def _calculate_similarity(markers1: Dict[str, Any], markers2: Dict[str, Any]) -> float:
        """Calculate similarity between two sets of markers."""
        # Simple implementation - could be made more sophisticated
        all_keys = set(markers1.keys()) | set(markers2.keys())
        if not all_keys:
            return 0.0
            
        matching_keys = 0
        for key in all_keys:
            if key in markers1 and key in markers2 and markers1[key] == markers2[key]:
                matching_keys += 1
                
        return matching_keys / len(all_keys)
