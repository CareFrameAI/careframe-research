import datetime
from typing import Dict, List, Set, Optional

class ValidatorRegistry:
    """Registry for managing validators and their credentials."""
    def __init__(self):
        self.validators = {}  # validator_id -> ValidatorInfo
        self.domains = {}     # domain -> set(validator_ids)
        self.roles = {}       # role -> set(validator_ids)
    
    def register_validator(self, validator_id: str, public_key: str, 
                          domains: List[str] = None, roles: List[str] = None) -> bool:
        """Register a new validator with domains and roles."""
        if validator_id in self.validators:
            return False
            
        self.validators[validator_id] = ValidatorInfo(
            validator_id, public_key, domains or [], roles or []
        )
        
        # Update domain indices
        for domain in (domains or []):
            if domain not in self.domains:
                self.domains[domain] = set()
            self.domains[domain].add(validator_id)
            
        # Update role indices
        for role in (roles or []):
            if role not in self.roles:
                self.roles[role] = set()
            self.roles[role].add(validator_id)
            
        return True
    
    def get_domain_validators(self, domain: str) -> List[str]:
        """Get all validators for a specific domain."""
        return list(self.domains.get(domain, set()))
    
    def get_role_validators(self, role: str) -> List[str]:
        """Get all validators with a specific role."""
        return list(self.roles.get(role, set()))
    
    def get_validator_public_key(self, validator_id: str) -> Optional[str]:
        """Get public key for a validator."""
        if validator_id not in self.validators:
            return None
        return self.validators[validator_id].public_key
    
    def update_validator_reputation(self, validator_id: str, change: int) -> bool:
        """Update reputation score for a validator."""
        if validator_id not in self.validators:
            return False
            
        self.validators[validator_id].update_reputation(change)
        return True

class ValidatorInfo:
    """Information about a validator."""
    def __init__(self, validator_id: str, public_key: str, 
                domains: List[str] = None, roles: List[str] = None):
        self.validator_id = validator_id
        self.public_key = public_key
        self.domains = domains or []
        self.roles = roles or []
        self.reputation = 100  # Initial reputation score
        self.validation_history = []
    
    def update_reputation(self, change: int) -> None:
        """Update reputation score."""
        self.reputation += change
        # Cap reputation between 0-100
        self.reputation = max(0, min(100, self.reputation))
    
    def add_validation(self, transaction_id: str, decision: str) -> None:
        """Record validation activity."""
        self.validation_history.append({
            "transaction_id": transaction_id,
            "decision": decision,
            "timestamp": str(datetime.datetime.now())
        })
