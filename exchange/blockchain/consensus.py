# consensus.py - Consensus mechanisms
from typing import Dict, List, Set, Optional
import datetime
from .core import Block, Blockchain
from .security import verify_signature
import json

class ConsensusBase:
    """Base class for consensus mechanisms."""
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
    
    def validate_block(self, block: Block, validator_id: str, signature: str) -> bool:
        """Base validation method to be implemented by specific consensus mechanisms."""
        raise NotImplementedError
    
    def is_block_ready(self, block: Block) -> bool:
        """Check if block has met consensus requirements."""
        raise NotImplementedError

class ProofOfAuthority(ConsensusBase):
    """Proof of Authority consensus - trusted validators validate blocks."""
    def __init__(self, blockchain: Blockchain, validators: Dict[str, str], required_validations: int = 3):
        super().__init__(blockchain)
        self.validators = validators  # validator_id -> public_key
        self.required_validations = required_validations
    
    def validate_block(self, block: Block, validator_id: str, signature: str) -> bool:
        """Validate a block with a validator's signature."""
        # Check if validator is authorized
        if validator_id not in self.validators:
            return False
        
        # IMPORTANT: We need to verify against the original block hash that was signed
        # not the current block hash which might have changed due to previous validations
        
        # Get original block hash to verify against (stored when the block was created)
        original_hash = getattr(block, 'original_hash', block.hash)
        
        # Verify signature against the original hash
        if not verify_signature(
            original_hash,
            signature,
            self.validators[validator_id]
        ):
            return False
        
        # Add validation to block
        block.add_validation(validator_id, signature)
        
        # If block is ready, add to blockchain
        if self.is_block_ready(block):
            block.finalize()
            self.blockchain.add_block(block)
            if hasattr(self.blockchain, 'pending_blocks') and block.hash in self.blockchain.pending_blocks:
                del self.blockchain.pending_blocks[block.hash]
            return True
        
        return True  # Validation successful, but block not yet ready
    
    def is_block_ready(self, block: Block) -> bool:
        """Check if block has enough validations."""
        # Count unique validators
        unique_validators = {v["validator_id"] for v in block.validations}
        return len(unique_validators) >= self.required_validations

class TimeLockValidation:
    """Time-locked validation ensures a waiting period before finalization."""
    def __init__(self, challenge_period_seconds: int = 259200):  # 3 days default
        self.challenge_period_seconds = challenge_period_seconds
        self.challenges = {}  # block_hash -> list of challenges
    
    def lock_block(self, block: Block) -> None:
        """Apply time lock to a block."""
        block.time_lock_start = datetime.datetime.now()
    
    def can_finalize(self, block: Block) -> bool:
        """Check if challenge period has passed."""
        if not hasattr(block, 'time_lock_start'):
            return False
        
        time_passed = (datetime.datetime.now() - block.time_lock_start).total_seconds()
        block_hash = block.hash
        
        # Block cannot be finalized if there are open challenges
        if block_hash in self.challenges and self.challenges[block_hash]:
            return False
            
        return time_passed >= self.challenge_period_seconds
    
    def add_challenge(self, block: Block, challenger_id: str, reason: str) -> bool:
        """Record a challenge against a block."""
        if not hasattr(block, 'time_lock_start'):
            return False
            
        time_passed = (datetime.datetime.now() - block.time_lock_start).total_seconds()
        
        # Can only challenge during challenge period
        if time_passed >= self.challenge_period_seconds:
            return False
            
        block_hash = block.hash
        
        if block_hash not in self.challenges:
            self.challenges[block_hash] = []
            
        self.challenges[block_hash].append({
            "challenger_id": challenger_id,
            "reason": reason,
            "timestamp": datetime.datetime.now()
        })
        
        return True

class TeamBasedValidator:
    """Validator implementation using teams and organizations structure"""
    
    def __init__(self, teams_db_url="http://localhost:5984", auth=("admin", "cfpwd")):
        self.teams_db_url = teams_db_url
        self.auth = auth
        self.validators = {}  # team_id -> validator_info
    
    def load_validators_from_teams(self):
        """Load validator data from teams database"""
        import requests
        
        try:
            response = requests.get(
                f"{self.teams_db_url}/teams/_all_docs?include_docs=true",
                auth=self.auth
            )
            
            if response.status_code == 200:
                teams = [row['doc'] for row in response.json()['rows'] if 'doc' in row]
                
                for team in teams:
                    team_id = team.get('_id')
                    if team_id:
                        self.validators[team_id] = {
                            'team_name': team.get('name', 'Unknown Team'),
                            'organization_id': team.get('organization_id', 'independent'),
                            'members': team.get('members', []),
                            'weight': 1.0  # Default weight, can be customized
                        }
                
                return True
            return False
        except Exception as e:
            print(f"Error loading validators from teams: {e}")
            return False
    
    def get_validator_for_user(self, email):
        """Get validator ID for a given user email"""
        for team_id, validator in self.validators.items():
            if any(member.get('email') == email for member in validator['members']):
                return team_id
        return None
    
    def is_valid_validator(self, validator_id):
        """Check if validator ID is valid"""
        return validator_id in self.validators
    
    def get_validator_members(self, validator_id):
        """Get team members for a validator"""
        if validator_id in self.validators:
            return self.validators[validator_id].get('members', [])
        return []
    
    def get_admin_validators(self):
        """Get IDs of all teams with admin members"""
        admin_teams = []
        for team_id, validator in self.validators.items():
            if any(member.get('is_admin', False) for member in validator['members']):
                admin_teams.append(team_id)
        return admin_teams
