# api.py - Public interface for blockchain interaction
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid

from .core import Blockchain, Block
from .consensus import ProofOfAuthority, TimeLockValidation
from .security import sign_data, verify_signature, generate_random_id
from .team_validator import TeamBasedValidator

class BlockchainAPI:
    """Public API for interacting with the blockchain."""
    def __init__(self, blockchain: Blockchain, consensus, validator_registry=None):
        self.blockchain = blockchain
        self.consensus = consensus
        self.validator_registry = validator_registry
        self.timelock = TimeLockValidation()
        self.team_validator = TeamBasedValidator()
        self.team_validator.load_validators_from_teams()
        
    # Add this method to BlockchainAPI class in blockchain/api.py
    def create_block_with_pending_transactions(self):
        """Create a block with any pending transactions and add it to the chain."""
        # Check if there are pending transactions
        if not hasattr(self.blockchain, 'pending_transactions') or not self.blockchain.pending_transactions:
            return None, "No pending transactions"
        
        try:
            # Create the block
            block_hash = self.create_block()
            
            # Return success
            return block_hash, f"Block created with {len(self.blockchain.pending_transactions)} transactions"
        except Exception as e:
            return None, f"Error creating block: {e}"

    # Add this method to the BlockchainAPI class in blockchain/api.py
    def sign_data(self, data, private_key):
        """
        Sign data with a private key to create a digital signature.
        
        Args:
            data: Data to sign (string or dict that will be converted to string)
            private_key: Private key to use for signing
            
        Returns:
            Signature string
        """
        try:
            # If data is a dict or other object, convert to JSON string
            if not isinstance(data, str):
                import json
                data = json.dumps(data)
                
            # For simple implementation, just use hashlib
            import hashlib
            import base64
            
            # Create signature by combining data and private key and hashing
            signature_data = f"{data}:{private_key}"
            signature_hash = hashlib.sha256(signature_data.encode()).digest()
            signature = base64.b64encode(signature_hash).decode()
            
            return signature
        except Exception as e:
            print(f"Error signing data: {e}")
            return "SIGNATURE_ERROR"

    def submit_hypothesis(self, hypothesis_data: Dict, submitter_id: str, private_key: str) -> str:
        """Submit a new hypothesis to the blockchain."""
        # Generate unique ID if not provided
        if 'id' not in hypothesis_data:
            hypothesis_data['id'] = generate_random_id("HYP-")
        
        hypothesis_id = hypothesis_data['id']
        
        # Create transaction - IMPORTANT: Make sure hypothesis_id is at top level
        transaction = {
            "type": "CREATE_HYPOTHESIS",
            "hypothesis_id": hypothesis_id,  # Explicitly at top level for easy lookup
            "data": hypothesis_data,
            "submitter_id": submitter_id,
            "timestamp": str(datetime.datetime.now())
        }
        
        # Sign transaction
        transaction_str = json.dumps(transaction, sort_keys=True)
        transaction["signature"] = sign_data(transaction_str, private_key)
        
        # Add to blockchain
        self.blockchain.add_transaction(transaction)
        
        return hypothesis_id
    
    def submit_evidence(self, hypothesis_id: str, evidence_data: Dict, 
                        submitter_id: str, private_key: str) -> str:
        """Add evidence to an existing hypothesis."""
        # Generate unique ID if not provided
        if 'id' not in evidence_data:
            evidence_data['id'] = generate_random_id("EVI-")
        
        # Create transaction
        transaction = {
            "type": "APPEND_EVIDENCE",
            "evidence_id": evidence_data['id'],
            "hypothesis_id": hypothesis_id,
            "data": evidence_data,
            "submitter_id": submitter_id,
            "timestamp": str(datetime.datetime.now())
        }
        
        # Sign transaction
        transaction_str = json.dumps(transaction, sort_keys=True)
        transaction["signature"] = sign_data(transaction_str, private_key)
        
        # Add to blockchain
        self.blockchain.add_transaction(transaction)
        
        return evidence_data['id']
    
    def submit_population_indicator(self, indicator_data: Dict, 
                                    submitter_id: str, private_key: str) -> str:
        """Add population indicator data to the blockchain."""
        # Generate unique ID if not provided
        if 'id' not in indicator_data:
            indicator_data['id'] = generate_random_id("POP-")
        
        # Create transaction
        transaction = {
            "type": "ADD_POPULATION_INDICATOR",
            "indicator_id": indicator_data['id'],
            "data": indicator_data,
            "submitter_id": submitter_id,
            "timestamp": str(datetime.datetime.now())
        }
        
        # Sign transaction
        transaction_str = json.dumps(transaction, sort_keys=True)
        transaction["signature"] = sign_data(transaction_str, private_key)
        
        # Add to blockchain
        self.blockchain.add_transaction(transaction)
        
        return indicator_data['id']
    
    def create_block(self) -> str:
        """Create a new block with pending transactions and return its hash."""
        # Return early if no pending transactions
        if not self.blockchain.pending_transactions:
            print("No pending transactions to create block")
            return None
            
        # Check if there are already pending blocks with these transactions
        # to avoid creating duplicate blocks
        pending_txs_hash = self._hash_pending_transactions()
        
        # If we already have pending blocks, check if any has the same transactions
        if hasattr(self.blockchain, 'pending_blocks') and self.blockchain.pending_blocks:
            for block_hash, block in self.blockchain.pending_blocks.items():
                block_txs_hash = self._hash_block_transactions(block)
                if block_txs_hash == pending_txs_hash:
                    print(f"Block with these transactions already exists: {block_hash[:10]}...")
                    return block_hash
        
        # Create a new block
        block = self.blockchain.create_block(self.blockchain.pending_transactions)
        
        # Return the block hash
        return block.hash
        
    def _hash_pending_transactions(self) -> str:
        """Create a hash of the pending transactions to detect duplicates."""
        import hashlib
        
        # Create a string representation of pending transactions
        tx_str = ""
        for tx in self.blockchain.pending_transactions:
            if hasattr(tx, 'tx_hash'):
                tx_str += tx.tx_hash
            elif isinstance(tx, dict) and 'tx_hash' in tx:
                tx_str += tx['tx_hash']
                
        # Hash the string
        return hashlib.sha256(tx_str.encode()).hexdigest()
        
    def _hash_block_transactions(self, block) -> str:
        """Create a hash of a block's transactions to detect duplicates."""
        import hashlib
        
        # Create a string representation of block transactions
        tx_str = ""
        for tx in block.transactions:
            if hasattr(tx, 'tx_hash'):
                tx_str += tx.tx_hash
            elif isinstance(tx, dict) and 'tx_hash' in tx:
                tx_str += tx['tx_hash']
                
        # Hash the string
        return hashlib.sha256(tx_str.encode()).hexdigest()
    
    def validate_block(self, block_hash: str, validator_id: str, private_key: str) -> bool:
        """Validate a block using team-based validator"""
        # Check if validator ID is valid
        is_admin = False
        
        # Check if this is an admin user - admins can validate any block
        if hasattr(self, 'admin_keys') and 'user_id' in self.admin_keys:
            admin_id = self.admin_keys.get('user_id')
            # If validator is admin, they can always validate
            if validator_id == admin_id:
                is_admin = True
                print(f"Admin user {validator_id} validating block")
        
        # For regular validators, check team membership
        if not is_admin and not self.team_validator.is_valid_validator(validator_id):
            # Try to refresh validators and check again
            self.team_validator.load_validators_from_teams()
            
            # Special case for solo users - if no validators exist, any user can validate
            solo_mode = len(self.team_validator.validators) == 0
            if solo_mode:
                print(f"Solo mode: User {validator_id} can validate without team")
            elif not self.team_validator.is_valid_validator(validator_id):
                print(f"Invalid validator {validator_id} - not authorized to validate blocks")
                return False
        
        # Get the block
        if block_hash not in self.blockchain.pending_blocks:
            print(f"Block {block_hash[:10]}... not found in pending blocks")
            return False
            
        block = self.blockchain.pending_blocks[block_hash]
        
        # Check if this validator has already validated this block
        if hasattr(block, 'validations'):
            for validation in block.validations:
                if validation.get('validator_id') == validator_id:
                    print(f"Validator {validator_id} already validated this block")
                    return True  # Already validated, consider it a success
        
        # Create validation record with appropriate team info
        team_name = "Admin" if is_admin else "Solo User"
        org_id = "system"
        
        if not is_admin and validator_id in self.team_validator.validators:
            team_name = self.team_validator.validators[validator_id]['team_name']
            org_id = self.team_validator.validators[validator_id]['organization_id']
        
        validation = {
            "validator_id": validator_id,
            "team_name": team_name,
            "organization_id": org_id,
            "timestamp": datetime.now().isoformat(),
            "signature": sign_data(block_hash, private_key)
        }
        
        # Add validation to block
        if not hasattr(block, 'validations'):
            block.validations = []
        block.validations.append(validation)
        print(f"Added validation from {validator_id} to block {block_hash[:10]}...")
        
        # Admin users only need 1 validation (their own)
        required_validations = 1 if is_admin else self.consensus.required_validations
        
        # Check if we have enough validations
        if len(block.validations) >= required_validations:
            print(f"Block {block_hash[:10]}... has enough validations ({len(block.validations)}/{required_validations}). Adding to blockchain...")
            
            # Check block index before adding
            expected_index = len(self.blockchain.chain)
            
            if block.index != expected_index:
                print(f"Block index mismatch: Block has index {block.index}, but chain expects {expected_index}")
                if is_admin:
                    # Admin can force correct block index
                    print(f"Admin forcing correct block index from {block.index} to {expected_index}")
                    block.index = expected_index
                else:
                    print(f"Cannot add block with incorrect index")
                    return False
            
            # Add the block to the blockchain
            try:
                # Call add_block to add the block to the chain
                added = self.blockchain.add_block(block)
                if added:
                    print(f"Block {block_hash[:10]}... added to blockchain")
                    
                    # Remove from pending blocks after adding to chain
                    if block_hash in self.blockchain.pending_blocks:
                        del self.blockchain.pending_blocks[block_hash]
                        print(f"Removed block {block_hash[:10]}... from pending blocks")
                    
                    # Remove transactions from pending after successful addition
                    tx_hashes = set()
                    for tx in block.transactions:
                        if hasattr(tx, 'tx_hash'):
                            tx_hashes.add(tx.tx_hash)
                        elif isinstance(tx, dict) and 'tx_hash' in tx:
                            tx_hashes.add(tx['tx_hash'])
                    
                    # Filter out transactions that were added to the block
                    self.blockchain.pending_transactions = [
                        tx for tx in self.blockchain.pending_transactions 
                        if not (hasattr(tx, 'tx_hash') and tx.tx_hash in tx_hashes) and
                           not (isinstance(tx, dict) and 'tx_hash' in tx and tx['tx_hash'] in tx_hashes)
                    ]
                    
                    return True
                else:
                    print(f"Failed to add block {block_hash[:10]}... to blockchain - chain rejected block")
                    return False
            except Exception as e:
                print(f"Error adding block to blockchain: {e}")
                return False
        
        print(f"Block {block_hash[:10]}... now has {len(block.validations)}/{required_validations} validations")
        return True
    
    def get_hypothesis_history(self, hypothesis_id: str) -> List[Dict]:
        """Get full history of a hypothesis and its evidence."""
        return self.blockchain.get_transaction_history(hypothesis_id)
    
    def challenge_block(self, block_hash: str, challenger_id: str, 
                       reason: str) -> bool:
        """Challenge a block during its challenge period."""
        # Find the block
        block = None
        for b in self.blockchain.chain:
            if b.hash == block_hash:
                block = b
                break
        
        if not block:
            return False
        
        # Submit challenge
        return self.timelock.add_challenge(block, challenger_id, reason)

    def submit_studies_manager_hypothesis(self, hypothesis, submitter_id, private_key):
        """Submit a hypothesis from StudiesManager to the blockchain with team info."""
        try:
            # Get team info if available
            team_info = None
            if hasattr(self, 'team_validator'):
                team_id = self.team_validator.get_validator_for_user(submitter_id)
                if team_id in self.team_validator.validators:
                    team_info = {
                        'team_id': team_id,
                        'team_name': self.team_validator.validators[team_id]['team_name'],
                        'organization_id': self.team_validator.validators[team_id]['organization_id']
                    }
            
            # Prepare transaction data
            transaction_data = {
                "type": "HYPOTHESIS",
                "hypothesis_id": hypothesis.get("id", "unknown"),
                "title": hypothesis.get("title", "Untitled Hypothesis"),
                "description": hypothesis.get("description", "No description"),
                "submitter": submitter_id,
                "team_info": team_info,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "status": hypothesis.get("status", "Unknown"),
                    "created_at": hypothesis.get("created_at", "Unknown")
                }
            }
            
            # Sign the transaction
            signature = self.sign_data(json.dumps(transaction_data), private_key)
            transaction_data["signature"] = signature
            
            # Submit the transaction
            self.blockchain.add_transaction(transaction_data)
            
            return transaction_data["hypothesis_id"]
            
        except Exception as e:
            print(f"Error submitting hypothesis to blockchain: {e}")
            return None

    def submit_studies_manager_evidence(self, hypothesis_id, evidence_data, submitter_id, private_key):
        """Submit evidence for a hypothesis to the blockchain."""
        try:
            # Fix datetime import
            from datetime import datetime
            
            # Prepare transaction data
            transaction_data = {
                "type": "EVIDENCE",
                "id": str(uuid.uuid4()),
                "hypothesis_id": hypothesis_id,
                "evidence_type": evidence_data.get("evidence_type", "Unknown"),
                "summary": evidence_data.get("summary", "No summary"),
                "confidence": evidence_data.get("confidence", 0.0),
                "submitter": submitter_id,
                "timestamp": datetime.now().isoformat(),
                "evidence_data": evidence_data
            }
            
            # Sign the transaction
            signature = self.sign_data(transaction_data, private_key)
            transaction_data["signature"] = signature
            
            # Submit the transaction
            self.blockchain.add_transaction(transaction_data)
            
            return transaction_data["id"]
            
        except Exception as e:
            print(f"Error submitting evidence to blockchain: {e}")
            return None

    def get_studies_manager_hypothesis_with_evidence(self, hypothesis_id: str) -> Dict:
        """Get a hypothesis with all its evidence in studies_manager format."""
        # Get all transactions related to this hypothesis
        all_transactions = self.blockchain.get_transaction_history(hypothesis_id)
        
        # Find the hypothesis creation transaction
        hypothesis_tx = None
        evidence_txs = []
        
        for item in all_transactions:
            tx = item["transaction"]
            if tx["type"] == "CREATE_HYPOTHESIS" and tx["hypothesis_id"] == hypothesis_id:
                hypothesis_tx = tx
            elif tx["type"] == "APPEND_EVIDENCE" and tx["hypothesis_id"] == hypothesis_id:
                evidence_txs.append(tx)
        
        if not hypothesis_tx:
            return None
        
        # Format the hypothesis with studies_manager format
        result = {
            "id": hypothesis_tx["hypothesis_id"],
            "title": hypothesis_tx.get("title", ""),
            "null_hypothesis": hypothesis_tx.get("null_hypothesis", ""),
            "alternative_hypothesis": hypothesis_tx.get("alternative_hypothesis", ""),
            "description": hypothesis_tx.get("description", ""),
            "created_at": hypothesis_tx.get("created_at", ""),
            "updated_at": hypothesis_tx.get("updated_at", ""),
            "status": hypothesis_tx.get("status", "untested"),
            "supporting_papers": hypothesis_tx.get("supporting_papers", []),
            "evidence": []
        }
        
        # Add all evidence
        for tx in evidence_txs:
            evidence = {
                "id": tx["evidence_id"],
                "evidence_type": tx["evidence_type"],
                "summary": tx["summary"],
                "timestamp": tx["timestamp"],
                "submitter_id": tx["submitter_id"],
                "confidence": tx.get("confidence")
            }
            
            # Add specific evidence data based on type
            if tx["evidence_type"] == "model" and "test_results" in tx.get("additional_data", {}):
                evidence["test_results"] = tx["additional_data"]["test_results"]
            elif tx["evidence_type"] == "literature" and "literature_evidence" in tx.get("additional_data", {}):
                evidence["literature_evidence"] = tx["additional_data"]["literature_evidence"]
            
            result["evidence"].append(evidence)
        
        return result

    def refresh_validators(self):
        """Refresh validator list from teams database"""
        return self.team_validator.load_validators_from_teams()

    def get_validation_teams(self):
        """Get all teams that can validate blocks"""
        return list(self.team_validator.validators.keys())
        
    def get_pending_blocks(self):
        """Get all pending blocks awaiting validation."""
        if hasattr(self.blockchain, 'pending_blocks'):
            return self.blockchain.pending_blocks
        return {}
        
    def get_pending_transactions(self):
        """Get all pending transactions not yet in a block."""
        if hasattr(self.blockchain, 'pending_transactions'):
            return self.blockchain.pending_transactions
        return []
        
    def get_chain(self):
        """Get the blockchain chain."""
        if hasattr(self.blockchain, 'chain'):
            return self.blockchain.chain
        return []

    def admin_force_mine_block(self, admin_id: str, private_key: str) -> Dict:
        """
        Admin-only method to force create and validate a block with pending transactions.
        
        Args:
            admin_id: ID of the admin user
            private_key: Admin's private key for signing
            
        Returns:
            Dict with status, message, and block_hash if successful
        """
        result = {
            "status": "error",
            "message": "",
            "block_hash": None
        }
        
        # Verify this is an admin user
        is_admin = False
        if hasattr(self, 'admin_keys') and self.admin_keys.get('user_id') == admin_id:
            is_admin = True
        
        if not is_admin:
            result["message"] = "Only admin users can force mine blocks"
            return result
            
        # Check if there are pending transactions
        if not self.blockchain.pending_transactions:
            result["message"] = "No pending transactions to mine"
            return result
            
        try:
            # Create a new block
            print(f"Admin {admin_id} forcing block creation with {len(self.blockchain.pending_transactions)} transactions")
            block_hash = self.create_block()
            
            if not block_hash:
                result["message"] = "Failed to create block"
                return result
                
            # Validate the block with admin privileges
            validation_result = self.validate_block(block_hash, admin_id, private_key)
            
            if validation_result:
                result["status"] = "success"
                result["message"] = f"Admin forced mining successful. Block {block_hash[:10]}... added to blockchain."
                result["block_hash"] = block_hash
            else:
                result["message"] = "Admin validation failed"
                
            return result
            
        except Exception as e:
            result["message"] = f"Error during force mining: {str(e)}"
            return result