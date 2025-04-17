# transactions/evidence.py - Updated to match studies_manager format

import datetime
import json
from typing import Dict, List, Optional

from exchange.blockchain.security import generate_random_id, sign_data, verify_signature


class EvidenceTransaction:
    """Transaction for adding evidence to a hypothesis."""
    def __init__(self, 
                 hypothesis_id: str, 
                 evidence_type: str,  # "model" or "literature"
                 summary: str,
                 submitter_id: str,
                 confidence: Optional[float] = None):
        self.transaction_type = "APPEND_EVIDENCE"
        self.evidence_id = generate_random_id("EVI-")
        self.hypothesis_id = hypothesis_id
        self.evidence_type = evidence_type
        self.summary = summary
        self.confidence = confidence
        self.submitter_id = submitter_id
        self.timestamp = datetime.datetime.now().isoformat()
        self.signature = None
        self.additional_data = {}
    
    def add_model_evidence(self, 
                          test_name: str, 
                          p_value: float, 
                          test_statistic: float,
                          dataset_name: Optional[str] = None,
                          alpha_level: float = 0.05) -> None:
        """Add model-based statistical evidence."""
        if self.evidence_type != "model":
            raise ValueError("Cannot add model evidence to a non-model evidence type")
            
        self.additional_data["test_results"] = {
            "test_name": test_name,
            "p_value": p_value,
            "test_statistic": test_statistic,
            "dataset_name": dataset_name,
            "alpha_level": alpha_level,
            "significant": p_value < alpha_level,
            "test_date": datetime.datetime.now().isoformat()
        }
    
    def add_literature_evidence(self, 
                              papers: List[Dict],
                              consensus_status: str,
                              effect_size_range: Optional[List[float]] = None) -> None:
        """Add literature-based evidence."""
        if self.evidence_type != "literature":
            raise ValueError("Cannot add literature evidence to a non-literature evidence type")
            
        self.additional_data["literature_evidence"] = {
            "papers": papers,
            "consensus_status": consensus_status,  # "confirmed", "rejected", "inconclusive", etc.
            "effect_size_range": effect_size_range,
            "paper_count": len(papers),
            "analysis_date": datetime.datetime.now().isoformat()
        }
    
    def to_dict(self, include_signature: bool = True) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "type": self.transaction_type,
            "evidence_id": self.evidence_id,
            "hypothesis_id": self.hypothesis_id,
            "evidence_type": self.evidence_type,
            "summary": self.summary,
            "submitter_id": self.submitter_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "additional_data": self.additional_data
        }
        
        if include_signature and self.signature:
            result["signature"] = self.signature
            
        return result
    
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