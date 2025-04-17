# storage.py - Persistence layer
import json
import os
import time
from typing import Optional, Dict, Any, List
from .core import Blockchain, Block
import datetime
import shutil

class BlockchainStorage:
    """Handles persistence of blockchain data."""
    def __init__(self, blockchain: Blockchain, storage_dir: str = "./blockchain_data"):
        self.blockchain = blockchain
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "backups"), exist_ok=True)
    
    def save_blockchain(self) -> bool:
        """Save the entire blockchain to disk."""
        try:
            # Convert blockchain to serializable format
            chain_data = []
            for block in self.blockchain.chain:
                block_dict = {
                    "index": block.index,
                    "timestamp": str(block.timestamp),
                    "transactions": block.transactions,
                    "previous_hash": block.previous_hash,
                    "hash": block.hash,
                    "validations": block.validations,
                    "finalized": block.finalized,
                    "finalization_time": str(block.finalization_time) if block.finalization_time else None
                }
                chain_data.append(block_dict)
            
            # Save chain data
            with open(os.path.join(self.storage_dir, "chain.json"), "w") as f:
                json.dump(chain_data, f, indent=2)
                
            # Save pending transactions
            with open(os.path.join(self.storage_dir, "pending_transactions.json"), "w") as f:
                json.dump(self.blockchain.pending_transactions, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving blockchain: {e}")
            return False
    
    def load_blockchain(self) -> bool:
        """Load blockchain from disk."""
        try:
            # Check if files exist
            chain_path = os.path.join(self.storage_dir, "chain.json")
            pending_path = os.path.join(self.storage_dir, "pending_transactions.json")
            
            if not os.path.exists(chain_path):
                return False
                
            # Load chain data
            with open(chain_path, "r") as f:
                chain_data = json.load(f)
                
            # Reconstruct blockchain
            new_chain = []
            for block_dict in chain_data:
                block = Block(
                    block_dict["index"],
                    block_dict["timestamp"],
                    block_dict["transactions"],
                    block_dict["previous_hash"]
                )
                block.hash = block_dict["hash"]
                block.validations = block_dict["validations"]
                block.finalized = block_dict["finalized"]
                if block_dict["finalization_time"]:
                    block.finalization_time = datetime.datetime.fromisoformat(
                        block_dict["finalization_time"].replace('Z', '+00:00')
                    )
                new_chain.append(block)
                
            # Replace current chain if valid
            if new_chain:
                # Verify chain integrity before replacing
                temp_blockchain = Blockchain()
                temp_blockchain.chain = new_chain
                if temp_blockchain.is_chain_valid():
                    self.blockchain.chain = new_chain
                else:
                    print("Loaded chain is invalid, keeping current chain")
                    return False
                
            # Load pending transactions if available
            if os.path.exists(pending_path):
                with open(pending_path, "r") as f:
                    self.blockchain.pending_transactions = json.load(f)
                    
            return True
        except Exception as e:
            print(f"Error loading blockchain: {e}")
            return False
    
    def backup_blockchain(self, backup_name: str = None) -> str:
        """Create a backup of the current blockchain state."""
        try:
            backup_name = backup_name or f"backup_{int(time.time())}"
            backup_dir = os.path.join(self.storage_dir, "backups", backup_name)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Save chain data to backup directory
            chain_data = []
            for block in self.blockchain.chain:
                block_dict = {
                    "index": block.index,
                    "timestamp": str(block.timestamp),
                    "transactions": block.transactions,
                    "previous_hash": block.previous_hash,
                    "hash": block.hash,
                    "validations": block.validations,
                    "finalized": block.finalized,
                    "finalization_time": str(block.finalization_time) if block.finalization_time else None
                }
                chain_data.append(block_dict)
                
            # Save backup
            with open(os.path.join(backup_dir, "chain.json"), "w") as f:
                json.dump(chain_data, f, indent=2)
                
            # Save pending transactions
            with open(os.path.join(backup_dir, "pending_transactions.json"), "w") as f:
                json.dump(self.blockchain.pending_transactions, f, indent=2)
                
            # Save backup info
            with open(os.path.join(backup_dir, "info.json"), "w") as f:
                info = {
                    "backup_time": str(datetime.datetime.now()),
                    "chain_length": len(self.blockchain.chain),
                    "pending_transactions": len(self.blockchain.pending_transactions)
                }
                json.dump(info, f, indent=2)
                
            return backup_dir
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """Restore blockchain from a backup."""
        try:
            backup_dir = os.path.join(self.storage_dir, "backups", backup_name)
            if not os.path.exists(backup_dir):
                print(f"Backup {backup_name} does not exist")
                return False
                
            # Create backup of current state before restoring
            current_backup = self.backup_blockchain("pre_restore_" + str(int(time.time())))
            
            # Copy backup files to main storage directory
            chain_src = os.path.join(backup_dir, "chain.json")
            chain_dest = os.path.join(self.storage_dir, "chain.json")
            shutil.copy2(chain_src, chain_dest)
            
            pending_src = os.path.join(backup_dir, "pending_transactions.json")
            if os.path.exists(pending_src):
                pending_dest = os.path.join(self.storage_dir, "pending_transactions.json")
                shutil.copy2(pending_src, pending_dest)
                
            # Load the restored data
            return self.load_blockchain()
        except Exception as e:
            print(f"Error restoring from backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List available backups with their information."""
        backups = []
        backup_dir = os.path.join(self.storage_dir, "backups")
        
        if not os.path.exists(backup_dir):
            return backups
            
        for entry in os.listdir(backup_dir):
            backup_path = os.path.join(backup_dir, entry)
            if os.path.isdir(backup_path):
                info_path = os.path.join(backup_path, "info.json")
                if os.path.exists(info_path):
                    try:
                        with open(info_path, "r") as f:
                            info = json.load(f)
                            info["name"] = entry
                            backups.append(info)
                    except:
                        # If info.json can't be read, still include backup in list
                        backups.append({
                            "name": entry,
                            "backup_time": str(datetime.datetime.fromtimestamp(
                                os.path.getctime(backup_path)
                            )),
                            "status": "info_unreadable"
                        })
                else:
                    # If info.json doesn't exist, still include backup in list
                    backups.append({
                        "name": entry,
                        "backup_time": str(datetime.datetime.fromtimestamp(
                            os.path.getctime(backup_path)
                        )),
                        "status": "no_info"
                    })
        
        # Sort by backup time (newest first)
        backups.sort(key=lambda x: x.get("backup_time", ""), reverse=True)
        return backups
    
    def export_to_file(self, file_path: str) -> bool:
        """Export entire blockchain to a single file."""
        try:
            export_data = {
                "chain": [],
                "pending_transactions": self.blockchain.pending_transactions,
                "export_time": str(datetime.datetime.now())
            }
            
            # Convert chain to serializable format
            for block in self.blockchain.chain:
                block_dict = {
                    "index": block.index,
                    "timestamp": str(block.timestamp),
                    "transactions": block.transactions,
                    "previous_hash": block.previous_hash,
                    "hash": block.hash,
                    "validations": block.validations,
                    "finalized": block.finalized,
                    "finalization_time": str(block.finalization_time) if block.finalization_time else None
                }
                export_data["chain"].append(block_dict)
                
            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error exporting blockchain: {e}")
            return False
    
    def import_from_file(self, file_path: str) -> bool:
        """Import blockchain from an export file."""
        try:
            if not os.path.exists(file_path):
                print(f"Import file {file_path} does not exist")
                return False
                
            # Backup current state
            self.backup_blockchain("pre_import_" + str(int(time.time())))
            
            with open(file_path, "r") as f:
                import_data = json.load(f)
                
            if "chain" not in import_data:
                print("Invalid import file: no chain data")
                return False
                
            # Reconstruct blockchain
            new_chain = []
            for block_dict in import_data["chain"]:
                block = Block(
                    block_dict["index"],
                    block_dict["timestamp"],
                    block_dict["transactions"],
                    block_dict["previous_hash"]
                )
                block.hash = block_dict["hash"]
                block.validations = block_dict["validations"]
                block.finalized = block_dict.get("finalized", False)
                if block_dict.get("finalization_time"):
                    block.finalization_time = datetime.datetime.fromisoformat(
                        block_dict["finalization_time"].replace('Z', '+00:00')
                    )
                new_chain.append(block)
                
            # Verify chain integrity
            temp_blockchain = Blockchain()
            temp_blockchain.chain = new_chain
            if not temp_blockchain.is_chain_valid():
                print("Imported chain is invalid")
                return False
                
            # Replace current chain
            self.blockchain.chain = new_chain
            
            # Import pending transactions if available
            if "pending_transactions" in import_data:
                self.blockchain.pending_transactions = import_data["pending_transactions"]
                
            return True
        except Exception as e:
            print(f"Error importing blockchain: {e}")
            return False
