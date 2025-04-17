from typing import Dict, List, Set, Optional
import datetime

class ExpertValidation:
    """Manage domain expert validation for scientific content."""
    def __init__(self, validator_registry):
        self.validator_registry = validator_registry
        self.pending_validations = {}  # transaction_id -> list of required domains
        self.completed_validations = {}  # transaction_id -> {domain: validator_decision}
    
    def assign_validators(self, transaction_id: str, domains: List[str]) -> Dict[str, List[str]]:
        """Assign appropriate validators for a transaction based on domains."""
        assigned = {}
        
        for domain in domains:
            validators = self.validator_registry.get_domain_validators(domain)
            if validators:
                assigned[domain] = validators
        
        # Record that this transaction needs validation
        self.pending_validations[transaction_id] = domains
        
        return assigned
    
    def record_validation(self, transaction_id: str, validator_id: str, 
                         domain: str, decision: str, comments: str = "") -> bool:
        """Record a domain expert's validation decision."""
        # Check if transaction requires validation
        if transaction_id not in self.pending_validations:
            return False
            
        # Check if domain is required
        if domain not in self.pending_validations[transaction_id]:
            return False
            
        # Check if validator is qualified for this domain
        domain_validators = self.validator_registry.get_domain_validators(domain)
        if validator_id not in domain_validators:
            return False
            
        # Initialize if needed
        if transaction_id not in self.completed_validations:
            self.completed_validations[transaction_id] = {}
            
        # Record validation
        self.completed_validations[transaction_id][domain] = {
            "validator_id": validator_id,
            "decision": decision,
            "comments": comments,
            "timestamp": str(datetime.datetime.now())
        }
        
        return True
    
    def is_fully_validated(self, transaction_id: str) -> bool:
        """Check if a transaction has been validated for all required domains."""
        if transaction_id not in self.pending_validations:
            return False
            
        if transaction_id not in self.completed_validations:
            return False
            
        # Check if all required domains have validations
        required = set(self.pending_validations[transaction_id])
        completed = set(self.completed_validations[transaction_id].keys())
        
        return required.issubset(completed)
    
    def get_validation_status(self, transaction_id: str) -> Dict:
        """Get validation status for a transaction."""
        if transaction_id not in self.pending_validations:
            return {"status": "not_required"}
            
        if transaction_id not in self.completed_validations:
            return {
                "status": "pending",
                "required": self.pending_validations[transaction_id],
                "completed": []
            }
            
        required = self.pending_validations[transaction_id]
        completed = list(self.completed_validations[transaction_id].keys())
        pending = [d for d in required if d not in completed]
        
        return {
            "status": "fully_validated" if not pending else "partially_validated",
            "required": required,
            "completed": completed,
            "pending": pending,
            "validations": self.completed_validations[transaction_id]
        }
