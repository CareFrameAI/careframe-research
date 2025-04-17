# transactions/hypothesis.py - Updated to match studies_manager format

from typing import Dict, Any, Optional, List
import datetime
import json
from ..security import sign_data, verify_signature, generate_random_id

class HypothesisTransaction:
    """Transaction for creating a scientific hypothesis matching studies_manager format."""
    def __init__(self, 
                 title: str, 
                 null_hypothesis: str,
                 alternative_hypothesis: str,
                 submitter_id: str, 
                 description: Optional[str] = None):
        self.transaction_type = "CREATE_HYPOTHESIS"
        self.hypothesis_id = generate_random_id("HYP-")
        self.title = title
        self.null_hypothesis = null_hypothesis
        self.alternative_hypothesis = alternative_hypothesis
        self.description = description
        self.submitter_id = submitter_id
        self.created_at = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now().isoformat()
        self.status = "untested"  # Initial status
        self.signature = None
        self.supporting_papers = []
    
    def to_dict(self, include_signature: bool = True) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "type": self.transaction_type,
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "description": self.description,
            "submitter_id": self.submitter_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "supporting_papers": self.supporting_papers
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