# studies_blockchain_bridge.py

from typing import Dict, Optional
from exchange.blockchain import (
    Blockchain, BlockchainAPI, ProofOfAuthority, 
    ValidatorRegistry, BlockchainStorage
)
from study_model.studies_manager import StudiesManager, Study

class StudiesBlockchainBridge:
    """Bridge between StudiesManager and Blockchain."""
    
    def __init__(self, blockchain_api: BlockchainAPI, studies_manager: StudiesManager):
        self.blockchain_api = blockchain_api
        self.studies_manager = studies_manager
    
    def push_hypothesis_to_blockchain(self, hypothesis_id: str, private_key: str) -> bool:
        """Push a hypothesis from studies_manager to the blockchain."""
        # Get the active study
        study = self.studies_manager.get_active_study()
        if not study:
            return False
        
        # Get the hypothesis
        hypothesis = self.studies_manager.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return False
        
        # Submit to blockchain
        blockchain_id = self.blockchain_api.submit_studies_manager_hypothesis(
            hypothesis,
            f"user_{study.id[-8:]}",  # Create submitter ID from study ID
            private_key
        )
        
        # Update the hypothesis with blockchain ID if different
        if blockchain_id != hypothesis_id:
            hypothesis["blockchain_id"] = blockchain_id
            self.studies_manager.update_hypothesis(hypothesis_id, {"blockchain_id": blockchain_id})
        
        return True
    
    def push_model_evidence_to_blockchain(self, hypothesis_id: str, test_results: Dict, private_key: str) -> bool:
        """Push model-based evidence to the blockchain."""
        # Get the active study
        study = self.studies_manager.get_active_study()
        if not study:
            return False
        
        # Get the hypothesis
        hypothesis = self.studies_manager.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return False
        
        # Create evidence data
        evidence_data = {
            "evidence_type": "model",
            "summary": f"Statistical test: {test_results.get('test_name', 'Unknown test')}",
            "confidence": 0.95,  # Default confidence
            "test_results": test_results
        }
        
        # Get blockchain ID if exists, otherwise use original ID
        blockchain_id = hypothesis.get("blockchain_id", hypothesis_id)
        
        # Submit to blockchain
        evidence_id = self.blockchain_api.submit_studies_manager_evidence(
            blockchain_id,
            evidence_data,
            f"user_{study.id[-8:]}",  # Create submitter ID from study ID
            private_key
        )
        
        return True
    
    def push_literature_evidence_to_blockchain(self, hypothesis_id: str, literature_evidence: Dict, private_key: str) -> bool:
        """Push literature-based evidence to the blockchain."""
        # Get the active study
        study = self.studies_manager.get_active_study()
        if not study:
            return False
        
        # Get the hypothesis
        hypothesis = self.studies_manager.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return False
        
        # Create evidence data
        evidence_data = {
            "evidence_type": "literature",
            "summary": f"Literature evidence: {literature_evidence.get('status', 'Unknown status')}",
            "confidence": literature_evidence.get("confidence", 0.8),
            "literature_evidence": literature_evidence
        }
        
        # Get blockchain ID if exists, otherwise use original ID
        blockchain_id = hypothesis.get("blockchain_id", hypothesis_id)
        
        # Submit to blockchain
        evidence_id = self.blockchain_api.submit_studies_manager_evidence(
            blockchain_id,
            evidence_data,
            f"user_{study.id[-8:]}",  # Create submitter ID from study ID
            private_key
        )
        
        return True
    
    def pull_hypothesis_from_blockchain(self, blockchain_id: str) -> Optional[str]:
        """Pull a hypothesis from blockchain into studies_manager."""
        # Get the hypothesis with all evidence
        blockchain_data = self.blockchain_api.get_studies_manager_hypothesis_with_evidence(blockchain_id)
        if not blockchain_data:
            return None
        
        # Format for studies_manager
        hypothesis_data = {
            "id": blockchain_data["id"],
            "title": blockchain_data["title"],
            "null_hypothesis": blockchain_data["null_hypothesis"],
            "alternative_hypothesis": blockchain_data["alternative_hypothesis"],
            "description": blockchain_data["description"],
            "created_at": blockchain_data["created_at"],
            "updated_at": blockchain_data["updated_at"],
            "status": blockchain_data["status"],
            "supporting_papers": blockchain_data["supporting_papers"],
            "blockchain_id": blockchain_id
        }
        
        # Add to studies_manager
        hypothesis_id = self.studies_manager.add_hypothesis_to_study(
            hypothesis_text=hypothesis_data["title"],
            hypothesis_data=hypothesis_data
        )
        
        # Process evidence
        for evidence in blockchain_data.get("evidence", []):
            if evidence["evidence_type"] == "model" and "test_results" in evidence:
                # Add statistical evidence
                self.studies_manager.update_hypothesis_with_test_results(
                    hypothesis_id,
                    evidence["test_results"]
                )
            elif evidence["evidence_type"] == "literature" and "literature_evidence" in evidence:
                # Add literature evidence
                self.studies_manager.update_hypothesis_with_literature(
                    hypothesis_id,
                    evidence["literature_evidence"]
                )
        
        return hypothesis_id
    
    def push_hypothesis_to_blockchain_simple(self, hypothesis_id: str) -> bool:
        """
        Simple version that just marks the hypothesis as being in the blockchain.
        This is a workaround for UI responsiveness while we fix the full implementation.
        """
        # Get the hypothesis
        hypothesis = self.studies_manager.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return False
        
        # Update the hypothesis with blockchain ID for display purposes
        # In a real implementation, this would happen after blockchain validation
        update_data = {"blockchain_id": hypothesis_id}  # Use same ID for simplicity
        self.studies_manager.update_hypothesis(hypothesis_id, update_data)
        
        return True