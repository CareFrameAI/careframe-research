# core.py - Foundational blockchain classes
import hashlib
import json
import datetime
from typing import List, Dict, Any, Optional

class Block:
    def __init__(self, index: int, timestamp, transactions: List[Dict], previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.validations = []
        self.finalized = False
        self.finalization_time = None
        self.hash = self.calculate_hash()
        self.original_hash = self.hash  # Save original hash for validation
        
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of block contents."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": str(self.timestamp),
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "validations": self.validations
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def add_validation(self, validator_id: str, signature: str) -> None:
        """Add a validation record to the block."""
        self.validations.append({
            "validator_id": validator_id,
            "signature": signature,
            "timestamp": str(datetime.datetime.now())
        })
        # Update hash after adding validation
        self.hash = self.calculate_hash()
    
    def finalize(self) -> None:
        """Mark block as finalized after validation requirements are met."""
        self.finalized = True
        self.finalization_time = datetime.datetime.now()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        
    def create_genesis_block(self) -> Block:
        """Create the first block in the chain."""
        return Block(0, datetime.datetime.now(), [{"type": "GENESIS"}], "0")
    
    def get_latest_block(self) -> Block:
        """Return the most recent block in the chain."""
        return self.chain[-1]
    
    def add_transaction(self, transaction):
        """Add a transaction to the list of pending transactions."""
        if not hasattr(self, 'pending_transactions'):
            self.pending_transactions = []
        
        # Add transaction to pending list
        self.pending_transactions.append(transaction)
        print(f"Added transaction: {transaction.get('type')} by {transaction.get('submitter', 'Unknown')}")
        print(f"Pending transactions: {len(self.pending_transactions)}")
        
        return True
    
    def create_block(self, transactions=None) -> Block:
        """Create a new block with current pending transactions."""
        # Use provided transactions if given, otherwise use pending_transactions
        block_transactions = transactions if transactions is not None else self.pending_transactions
        
        if not block_transactions:
            return None
        
        latest_block = self.get_latest_block()
        new_block = Block(
            index=latest_block.index + 1,
            timestamp=datetime.datetime.now(),
            transactions=block_transactions.copy(),
            previous_hash=latest_block.hash
        )
        
        # Store the block in pending_blocks
        if not hasattr(self, 'pending_blocks'):
            self.pending_blocks = {}
        self.pending_blocks[new_block.hash] = new_block
        
        return new_block
    
    def add_block(self, block: Block) -> bool:
        """Add a validated block to the chain."""
        if block.index != self.get_latest_block().index + 1:
            return False
        
        if block.previous_hash != self.get_latest_block().hash:
            return False
        
        # In production, validate block before adding
        self.chain.append(block)
        # Clear only the transactions that were added to this block
        self.pending_transactions = [tx for tx in self.pending_transactions 
                                    if tx not in block.transactions]
        return True
    
    def is_chain_valid(self) -> bool:
        """Validate integrity of the entire blockchain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Verify hash
            if current.hash != current.calculate_hash():
                return False
                
            # Verify chain linkage
            if current.previous_hash != previous.hash:
                return False
                
        return True
    
    def get_transaction_history(self, identifier: str) -> List[Dict]:
        """Get all transactions related to a specific identifier."""
        history = []
        
        # Debug print to see what we're searching for
        print(f"Searching for transactions with ID: {identifier}")
        
        # First, print all transactions for debugging
        for i, block in enumerate(self.chain):
            print(f"Block {i} has {len(block.transactions) if isinstance(block.transactions, list) else 'N/A'} transactions")
            if i > 0 and isinstance(block.transactions, list):  # Skip genesis
                for j, tx in enumerate(block.transactions):
                    if isinstance(tx, dict):
                        print(f"  Tx {j}: type={tx.get('type')}, hypothesis_id={tx.get('hypothesis_id', 'None')}")
        
        # Search all blocks
        for block in self.chain:
            if block.index == 0:  # Skip genesis block
                continue
            
            if not isinstance(block.transactions, list):
                continue
            
            for tx in block.transactions:
                if not isinstance(tx, dict):
                    continue
                
                # Direct match on hypothesis_id field
                if tx.get('hypothesis_id') == identifier:
                    history.append({
                        "block_index": block.index,
                        "block_hash": block.hash,
                        "transaction": tx,
                        "timestamp": block.timestamp,
                        "validations": block.validations
                    })
        
        return history
