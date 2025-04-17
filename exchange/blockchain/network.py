# network.py - Peer-to-peer networking for blockchain distribution
import socket
import threading
import json
import time
from typing import Dict, List, Callable, Any, Optional
import requests
from .core import Blockchain, Block
from .security import verify_signature

class Node:
    """A node in the blockchain peer-to-peer network."""
    def __init__(self, host: str, port: int, blockchain: Blockchain):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.peers = {}  # {node_id: {"host": host, "port": port}}
        self.node_id = f"{host}:{port}"
        self.running = False
        self.server_socket = None
        self.handlers = {
            "get_chain": self._handle_get_chain,
            "get_pending": self._handle_get_pending,
            "new_block": self._handle_new_block,
            "new_transaction": self._handle_new_transaction,
            "add_peer": self._handle_add_peer,
        }
        
    def start(self) -> bool:
        """Start the node server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            # Start listener thread
            threading.Thread(target=self._listen_for_connections, daemon=True).start()
            print(f"Node started at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Error starting node: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the node server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
    
    def _listen_for_connections(self) -> None:
        """Listen for incoming connections."""
        while self.running:
            try:
                client, address = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
            except Exception as e:
                if self.running:  # Only log if not deliberately stopped
                    print(f"Connection error: {e}")
    
    def _handle_client(self, client_socket) -> None:
        """Handle incoming client connection."""
        try:
            # Receive data from client
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                
            if data:
                message = json.loads(data.decode('utf-8'))
                response = self._process_message(message)
                client_socket.sendall(json.dumps(response).encode('utf-8'))
                
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: Dict) -> Dict:
        """Process incoming message."""
        message_type = message.get("type")
        if message_type in self.handlers:
            return self.handlers[message_type](message)
        else:
            return {"status": "error", "message": "Unknown message type"}
    
    def _handle_get_chain(self, message: Dict) -> Dict:
        """Handle request for blockchain data."""
        chain_data = []
        for block in self.blockchain.chain:
            block_dict = {
                "index": block.index,
                "timestamp": str(block.timestamp),
                "transactions": block.transactions,
                "previous_hash": block.previous_hash,
                "hash": block.hash,
                "validations": block.validations,
            }
            chain_data.append(block_dict)
            
        return {"status": "success", "chain": chain_data}
    
    def _handle_get_pending(self, message: Dict) -> Dict:
        """Handle request for pending transactions."""
        return {"status": "success", "pending": self.blockchain.pending_transactions}
    
    def _handle_new_block(self, message: Dict) -> Dict:
        """Handle new block announcement."""
        block_data = message.get("block")
        if not block_data:
            return {"status": "error", "message": "No block data"}
            
        # Recreate block
        try:
            new_block = Block(
                block_data["index"],
                block_data["timestamp"],
                block_data["transactions"],
                block_data["previous_hash"]
            )
            new_block.hash = block_data["hash"]
            new_block.validations = block_data["validations"]
            
            # Try to add to blockchain
            if self.blockchain.add_block(new_block):
                return {"status": "success", "message": "Block added"}
            else:
                return {"status": "error", "message": "Failed to add block"}
        except Exception as e:
            return {"status": "error", "message": f"Error processing block: {e}"}
    
    def _handle_new_transaction(self, message: Dict) -> Dict:
        """Handle new transaction announcement."""
        transaction = message.get("transaction")
        if not transaction:
            return {"status": "error", "message": "No transaction data"}
            
        # Add to pending transactions
        self.blockchain.add_transaction(transaction)
        return {"status": "success", "message": "Transaction added"}
    
    def _handle_add_peer(self, message: Dict) -> Dict:
        """Handle peer registration."""
        peer_host = message.get("host")
        peer_port = message.get("port")
        if not peer_host or not peer_port:
            return {"status": "error", "message": "Missing peer information"}
            
        peer_id = f"{peer_host}:{peer_port}"
        self.peers[peer_id] = {"host": peer_host, "port": peer_port}
        return {"status": "success", "message": "Peer added", "node_id": self.node_id}
    
    # -- Client methods for sending requests to other nodes --
    
    def connect_to_peer(self, host: str, port: int) -> bool:
        """Connect to a peer node."""
        peer_id = f"{host}:{port}"
        if peer_id in self.peers:
            return True  # Already connected
            
        try:
            # Send add_peer request
            response = self._send_message(host, port, {
                "type": "add_peer",
                "host": self.host,
                "port": self.port
            })
            
            if response and response.get("status") == "success":
                self.peers[peer_id] = {"host": host, "port": port}
                return True
            return False
        except Exception as e:
            print(f"Error connecting to peer {host}:{port}: {e}")
            return False
    
    def broadcast_transaction(self, transaction: Dict) -> int:
        """Broadcast transaction to all peers."""
        success_count = 0
        message = {
            "type": "new_transaction",
            "transaction": transaction
        }
        
        for peer_id, peer in self.peers.items():
            try:
                response = self._send_message(peer["host"], peer["port"], message)
                if response and response.get("status") == "success":
                    success_count += 1
            except Exception as e:
                print(f"Error broadcasting to {peer_id}: {e}")
                
        return success_count
    
    def broadcast_block(self, block: Block) -> int:
        """Broadcast new block to all peers."""
        success_count = 0
        block_dict = {
            "index": block.index,
            "timestamp": str(block.timestamp),
            "transactions": block.transactions,
            "previous_hash": block.previous_hash,
            "hash": block.hash,
            "validations": block.validations,
        }
        
        message = {
            "type": "new_block",
            "block": block_dict
        }
        
        for peer_id, peer in self.peers.items():
            try:
                response = self._send_message(peer["host"], peer["port"], message)
                if response and response.get("status") == "success":
                    success_count += 1
            except Exception as e:
                print(f"Error broadcasting to {peer_id}: {e}")
                
        return success_count
    
    def sync_with_network(self) -> bool:
        """Sync blockchain with peers."""
        # Find the longest valid chain in the network
        longest_chain = None
        longest_length = len(self.blockchain.chain)
        
        for peer_id, peer in self.peers.items():
            try:
                # Get peer's chain
                response = self._send_message(peer["host"], peer["port"], {"type": "get_chain"})
                if response and response.get("status") == "success":
                    chain_data = response.get("chain", [])
                    
                    # Check if chain is longer and valid
                    if len(chain_data) > longest_length:
                        # TODO: Validate chain before accepting
                        longest_chain = chain_data
                        longest_length = len(chain_data)
            except Exception as e:
                print(f"Error syncing with {peer_id}: {e}")
        
        # Replace our chain if we found a longer valid one
        if longest_chain:
            # TODO: Convert chain_data back to Block objects and replace
            # For now, placeholder:
            # self.blockchain.chain = recreated_chain
            return True
            
        return False
    
    def _send_message(self, host: str, port: int, message: Dict) -> Optional[Dict]:
        """Send message to another node."""
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((host, port))
            client.sendall(json.dumps(message).encode('utf-8'))
            
            # Wait for response
            data = b""
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk
                
            client.close()
            
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            print(f"Error sending message to {host}:{port}: {e}")
            return None


class HTTPNode:
    """A node that communicates with RESTful HTTP API instead of raw sockets."""
    def __init__(self, host: str, port: int, blockchain: Blockchain):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.node_id = f"{host}:{port}"
        self.peers = {}  # {node_id: {"url": url}}
        
    def register_with_node(self, node_url: str) -> bool:
        """Register with another node."""
        try:
            response = requests.post(f"{node_url}/nodes/register", json={
                "nodes": [f"http://{self.host}:{self.port}"]
            })
            
            if response.status_code == 200:
                # Add this node to our peers
                parsed_url = node_url.rstrip('/')
                node_id = parsed_url.split('//')[-1]
                self.peers[node_id] = {"url": parsed_url}
                return True
            return False
        except Exception as e:
            print(f"Error registering with node {node_url}: {e}")
            return False
    
    def broadcast_transaction(self, transaction: Dict) -> int:
        """Broadcast transaction to all peers."""
        success_count = 0
        
        for peer_id, peer in self.peers.items():
            try:
                response = requests.post(f"{peer['url']}/transactions/new", 
                                      json={"transaction": transaction})
                
                if response.status_code == 200:
                    success_count += 1
            except Exception as e:
                print(f"Error broadcasting to {peer_id}: {e}")
                
        return success_count
    
    def broadcast_block(self, block: Block) -> int:
        """Broadcast new block to all peers."""
        success_count = 0
        block_dict = {
            "index": block.index,
            "timestamp": str(block.timestamp),
            "transactions": block.transactions,
            "previous_hash": block.previous_hash,
            "hash": block.hash,
            "validations": block.validations,
        }
        
        for peer_id, peer in self.peers.items():
            try:
                response = requests.post(f"{peer['url']}/blocks/new", 
                                      json={"block": block_dict})
                
                if response.status_code == 200:
                    success_count += 1
            except Exception as e:
                print(f"Error broadcasting to {peer_id}: {e}")
                
        return success_count
    
    def sync_with_network(self) -> bool:
        """Sync blockchain with peers."""
        # Find the longest valid chain in the network
        longest_chain = None
        longest_length = len(self.blockchain.chain)
        
        for peer_id, peer in self.peers.items():
            try:
                response = requests.get(f"{peer['url']}/chain")
                
                if response.status_code == 200:
                    chain_data = response.json().get("chain", [])
                    
                    # Check if chain is longer and valid
                    if len(chain_data) > longest_length:
                        # TODO: Validate chain before accepting
                        longest_chain = chain_data
                        longest_length = len(chain_data)
            except Exception as e:
                print(f"Error syncing with {peer_id}: {e}")
        
        # Replace our chain if we found a longer valid one
        if longest_chain:
            # TODO: Convert chain_data back to Block objects and replace
            return True
            
        return False
