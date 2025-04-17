import hashlib
import json
from time import time
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime
import uuid
import os

class EvidenceClaim:
    """
    Represents a single evidence claim that can be stored on the blockchain.
    """
    def __init__(
        self,
        claim_id: str,
        claim_text: str,
        claim_type: str,
        evidence_type: str,
        source_id: str,
        source_type: str,
        confidence_score: float,
        supporting_data: Dict[str, Any],
        metadata: Dict[str, Any],
        created_at: str,
        created_by: str
    ):
        self.claim_id = claim_id
        self.claim_text = claim_text
        self.claim_type = claim_type  # e.g., "causal", "associative", "comparative"
        self.evidence_type = evidence_type  # e.g., "statistical", "observational", "experimental"
        self.source_id = source_id  # ID of the source (e.g., study_id, paper_id)
        self.source_type = source_type  # e.g., "primary_study", "meta_analysis", "systematic_review"
        self.confidence_score = confidence_score  # 0.0 to 1.0
        self.supporting_data = supporting_data  # Statistical results, p-values, etc.
        self.metadata = metadata  # Any additional contextual information
        self.created_at = created_at
        self.created_by = created_by  # Researcher ID or name

class StudyComponent:
    """Base class for study components that can be stored on the blockchain."""
    def __init__(self, component_id: str, component_type: str, details: Dict[str, Any]):
        self.component_id = component_id
        self.component_type = component_type
        self.details = details
        self.version = "1.0"
        self.created_at = datetime.now().isoformat()
        self.last_modified = self.created_at

class LiteratureReview(StudyComponent):
    def __init__(self, component_id: str, thematic_sections: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """
        Represents a literature review with thematic sections.
        
        Args:
            component_id: Unique identifier for this component
            thematic_sections: List of sections each with title, body, citations, etc.
            metadata: Additional metadata like review methodology, databases searched, etc.
        """
        details = {
            "thematic_sections": thematic_sections,
            "metadata": metadata or {}
        }
        super().__init__(component_id, "literature_review", details)

class ProtocolSection:
    """Represents a section within a study protocol."""
    def __init__(
        self,
        section_id: str,
        title: str,
        body: str,
        version: str,
        timestamp: str,
        author: str,
        contributors: List[str],
        contributions: Dict[str, List[Any]],
        revision_history: List[Dict[str, Any]] = None
    ):
        self.section_id = section_id
        self.title = title
        self.body = body
        self.version = version
        self.timestamp = timestamp
        self.author = author
        self.contributors = contributors
        self.contributions = contributions
        self.revision_history = revision_history or []

class StudyProtocol(StudyComponent):
    def __init__(
        self,
        component_id: str,
        sections: List[ProtocolSection],
        protocol_metadata: Dict[str, Any] = None
    ):
        """
        Represents a complete study protocol with versioned sections.
        
        Args:
            component_id: Unique identifier for this component
            sections: List of protocol sections
            protocol_metadata: Additional metadata about the protocol
        """
        # Convert sections to dicts for JSON serialization
        sections_dict = [section.__dict__ for section in sections]
        
        details = {
            "sections": sections_dict,
            "metadata": protocol_metadata or {},
            "version_history": []  # Track protocol versions
        }
        super().__init__(component_id, "study_protocol", details)

class DataSource:
    """Represents a data source used in a study."""
    def __init__(
        self,
        source_id: str,
        name: str,
        data_source_type: str,
        source_details: Dict[str, Any],
        rows: int,
        columns: int,
        column_names: List[str],
        column_metadata: Dict[str, Any],
        column_types: Dict[str, str],
        privacy_level: str = "standard",
        access_restrictions: List[str] = None,
        data_provenance: Dict[str, Any] = None
    ):
        self.source_id = source_id
        self.name = name
        self.data_source_type = data_source_type
        self.source_details = source_details
        self.rows = rows
        self.columns = columns
        self.column_names = column_names
        self.column_metadata = column_metadata
        self.column_types = column_types
        self.privacy_level = privacy_level
        self.access_restrictions = access_restrictions or []
        self.data_provenance = data_provenance or {}
        self.created_at = datetime.now().isoformat()

class StatisticalTest:
    """Represents a statistical test performed during analysis."""
    def __init__(
        self,
        test_id: str,
        test_name: str,
        test_type: str,
        variables: Dict[str, List[str]],  # e.g. {"outcome": ["var1"], "predictor": ["var2"]}
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        assumptions_checked: Dict[str, bool],
        analysis_details: str,
        software_used: Dict[str, str],
        created_at: str,
        executed_by: str
    ):
        self.test_id = test_id
        self.test_name = test_name
        self.test_type = test_type
        self.variables = variables
        self.parameters = parameters
        self.results = results
        self.assumptions_checked = assumptions_checked
        self.analysis_details = analysis_details
        self.software_used = software_used
        self.created_at = created_at
        self.executed_by = executed_by

class StudyAnalysis(StudyComponent):
    def __init__(
        self,
        component_id: str,
        data_sources: List[DataSource],
        intervention_variables: List[str],
        outcome_variables: List[str],
        confounding_variables: List[str],
        statistical_tests: List[StatisticalTest],
        analysis_plan: Dict[str, Any] = None,
        preregistered: bool = False,
        analysis_metadata: Dict[str, Any] = None
    ):
        """
        Represents a study analysis component with detailed statistical tests.
        
        Args:
            component_id: Unique identifier for this component
            data_sources: List of data sources used
            intervention_variables: Variables representing interventions
            outcome_variables: Variables representing outcomes
            confounding_variables: Variables that may confound results
            statistical_tests: List of statistical tests performed
            analysis_plan: Details about the analysis plan
            preregistered: Whether the analysis was preregistered
            analysis_metadata: Additional metadata
        """
        # Convert objects to dicts for JSON serialization
        data_sources_dict = [ds.__dict__ for ds in data_sources]
        statistical_tests_dict = [test.__dict__ for test in statistical_tests]
        
        details = {
            "data_sources": data_sources_dict,
            "intervention_variables": intervention_variables,
            "outcome_variables": outcome_variables,
            "confounding_variables": confounding_variables,
            "statistical_tests": statistical_tests_dict,
            "analysis_plan": analysis_plan or {},
            "preregistered": preregistered,
            "metadata": analysis_metadata or {}
        }
        super().__init__(component_id, "study_analysis", details)

class StudyMetadata:
    """Comprehensive metadata about a study."""
    def __init__(
        self,
        study_id: str,
        title: str,
        authors: List[str],
        institutions: List[str],
        funding_sources: List[Dict[str, str]],
        publication_status: str,
        publication_date: Optional[str] = None,
        registration_id: Optional[str] = None,
        registration_date: Optional[str] = None,
        population_indicators: Dict[str, Any] = None,
        geographic_data: Dict[str, Any] = None,
        temporal_coverage: Dict[str, Any] = None,
        ethical_approvals: List[Dict[str, Any]] = None,
        keywords: List[str] = None,
        related_studies: List[str] = None
    ):
        self.study_id = study_id
        self.title = title
        self.authors = authors
        self.institutions = institutions
        self.funding_sources = funding_sources
        self.publication_status = publication_status
        self.publication_date = publication_date
        self.registration_id = registration_id
        self.registration_date = registration_date
        self.population_indicators = population_indicators or {}
        self.geographic_data = geographic_data or {}
        self.temporal_coverage = temporal_coverage or {}
        self.ethical_approvals = ethical_approvals or []
        self.keywords = keywords or []
        self.related_studies = related_studies or []
        self.created_at = datetime.now().isoformat()

class StudyEvidence:
    """Complete representation of study evidence for blockchain storage."""
    def __init__(
        self,
        evidence_id: str,
        study_id: str,
        title: str,
        description: str,
        intervention_text: str,
        outcome_results: Dict[str, Any],
        evidence_claims: List[EvidenceClaim],
        study_components: List[StudyComponent],
        metadata: StudyMetadata,
        evidence_quality: Dict[str, Any] = None,
        certainty_rating: str = "moderate",
        external_validations: List[Dict[str, Any]] = None
    ):
        self.evidence_id = evidence_id
        self.study_id = study_id
        self.title = title
        self.description = description
        self.intervention_text = intervention_text
        self.outcome_results = outcome_results
        
        # Convert evidence claims and study components to dicts for JSON serialization
        self.evidence_claims = [claim.__dict__ for claim in evidence_claims]
        self.study_components = [component.__dict__ for component in study_components]
        
        # Convert metadata to dict
        self.metadata = metadata.__dict__
        
        self.evidence_quality = evidence_quality or {}
        self.certainty_rating = certainty_rating
        self.external_validations = external_validations or []
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at

class Block:
    """Blockchain block containing study evidence."""
    def __init__(
        self,
        index: int,
        timestamp: float,
        evidence: StudyEvidence,
        previous_hash: str,
        block_type: str = "evidence",
        creator_id: str = None,
        signature: str = None,
        additional_metadata: Dict[str, Any] = None
    ):
        self.index = index
        self.timestamp = timestamp
        self.evidence = evidence.__dict__  # Store StudyEvidence as a dict
        self.previous_hash = previous_hash
        self.block_type = block_type
        self.creator_id = creator_id
        self.signature = signature
        self.additional_metadata = additional_metadata or {}
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Creates a SHA-256 hash of the block's dictionary."""
        # Remove the hash field from the dictionary to avoid circular reference
        block_copy = self.__dict__.copy()
        if 'hash' in block_copy:
            del block_copy['hash']
            
        block_string = json.dumps(block_copy, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @classmethod
    def from_dict(cls, block_data: Dict[str, Any]) -> 'Block':
        """Reconstructs a Block object from its dictionary representation."""
        # Create an uninitialized Block
        block = cls.__new__(cls)
        
        # Populate all attributes from the dictionary
        for key, value in block_data.items():
            setattr(block, key, value)
            
        return block
    
    def verify_hash(self) -> bool:
        """Verifies that the stored hash matches the calculated hash."""
        return self.hash == self.calculate_hash()
    
    def verify_signature(self, public_key=None) -> bool:
        """
        Verify the digital signature of the block (placeholder method).
        In a real implementation, this would use cryptographic signature verification.
        """
        # This is a simplified placeholder
        return self.signature is not None

class Blockchain:
    """
    Enhanced blockchain for storing and retrieving study evidence with improved
    querying capabilities and data integrity features.
    """
    def __init__(self):
        self.chain: List[Block] = [self.create_genesis_block()]
        self.block_index: Dict[str, int] = {}  # Maps evidence_id to block index
        self.claim_index: Dict[str, Set[int]] = {}  # Maps claim_id to block indices
        self.component_index: Dict[str, Set[int]] = {}  # Maps component_id to block indices
        self.study_index: Dict[str, Set[int]] = {}  # Maps study_id to block indices
        self.pending_transactions: List[StudyEvidence] = []  # For batch processing
        
        # Build indices for the genesis block
        self._update_indices(self.chain[0])

    def create_genesis_block(self) -> Block:
        """Creates the genesis block with minimal placeholder data."""
        genesis_metadata = StudyMetadata(
            study_id="genesis",
            title="Genesis Block",
            authors=["System"],
            institutions=["System"],
            funding_sources=[],
            publication_status="system"
        )
        
        genesis_evidence = StudyEvidence(
            evidence_id="genesis",
            study_id="genesis",
            title="Genesis Block",
            description="Initial block in the evidence blockchain",
            intervention_text="",
            outcome_results={},
            evidence_claims=[],
            study_components=[],
            metadata=genesis_metadata
        )
        
        return Block(
            index=0,
            timestamp=time(),
            evidence=genesis_evidence,
            previous_hash="0",
            block_type="genesis",
            creator_id="system"
        )

    def last_block(self) -> Block:
        """Returns the most recent block in the chain."""
        return self.chain[-1]

    def add_block(self, new_evidence: StudyEvidence, creator_id: str = None, signature: str = None) -> str:
        """
        Adds a new block to the blockchain.
        
        Args:
            new_evidence: The evidence to store in the block
            creator_id: Identifier of the block creator
            signature: Digital signature for authentication
            
        Returns:
            The hash of the new block
        """
        # Create the new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time(),
            evidence=new_evidence,
            previous_hash=self.last_block().hash,
            creator_id=creator_id,
            signature=signature
        )
        
        # Add the block to the chain
        self.chain.append(new_block)
        
        # Update indices
        self._update_indices(new_block)
        
        return new_block.hash

    def _update_indices(self, block: Block) -> None:
        """Updates the various indices with information from a block."""
        if not isinstance(block.evidence, dict):
            # Convert to dict if it's an object
            evidence_dict = block.evidence.__dict__
        else:
            evidence_dict = block.evidence
            
        # Extract IDs for indexing
        evidence_id = evidence_dict.get('evidence_id')
        study_id = evidence_dict.get('study_id')
        
        # Index the block by evidence_id
        if evidence_id:
            self.block_index[evidence_id] = block.index
            
        # Index the block by study_id
        if study_id:
            if study_id not in self.study_index:
                self.study_index[study_id] = set()
            self.study_index[study_id].add(block.index)
            
        # Index by claim_id
        for claim in evidence_dict.get('evidence_claims', []):
            if isinstance(claim, dict):
                claim_id = claim.get('claim_id')
                if claim_id:
                    if claim_id not in self.claim_index:
                        self.claim_index[claim_id] = set()
                    self.claim_index[claim_id].add(block.index)
            
        # Index by component_id
        for component in evidence_dict.get('study_components', []):
            if isinstance(component, dict):
                component_id = component.get('component_id')
                if component_id:
                    if component_id not in self.component_index:
                        self.component_index[component_id] = set()
                    self.component_index[component_id].add(block.index)

    def get_block_by_index(self, index: int) -> Optional[Block]:
        """Retrieves a block by its index in the chain."""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None

    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """Retrieves a block by its hash."""
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None

    def get_evidence_by_id(self, evidence_id: str) -> Optional[Dict]:
        """Retrieves evidence by its ID."""
        if evidence_id in self.block_index:
            block = self.get_block_by_index(self.block_index[evidence_id])
            if block:
                return block.evidence
        return None

    def get_blocks_by_study_id(self, study_id: str) -> List[Block]:
        """Retrieves all blocks related to a specific study."""
        if study_id not in self.study_index:
            return []
            
        return [self.chain[idx] for idx in self.study_index[study_id] if 0 <= idx < len(self.chain)]

    def get_blocks_by_claim_id(self, claim_id: str) -> List[Block]:
        """Retrieves all blocks containing a specific claim."""
        if claim_id not in self.claim_index:
            return []
            
        return [self.chain[idx] for idx in self.claim_index[claim_id] if 0 <= idx < len(self.chain)]

    def get_blocks_by_component_id(self, component_id: str) -> List[Block]:
        """Retrieves all blocks containing a specific component."""
        if component_id not in self.component_index:
            return []
            
        return [self.chain[idx] for idx in self.component_index[component_id] if 0 <= idx < len(self.chain)]

    def get_all_evidence(self) -> List[Dict]:
        """Returns all evidence records (excluding the genesis block)."""
        return [block.evidence for block in self.chain[1:]]

    def get_claims_by_type(self, claim_type: str) -> List[Dict]:
        """
        Retrieves all claims of a specific type.
        
        Args:
            claim_type: The type of claim to filter by (e.g., "causal", "associative")
            
        Returns:
            List of matching claims with block information
        """
        matching_claims = []
        
        for block in self.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            
            if not isinstance(evidence, dict):
                continue
                
            for claim in evidence.get('evidence_claims', []):
                if isinstance(claim, dict) and claim.get('claim_type') == claim_type:
                    # Add block context to the claim
                    enriched_claim = claim.copy()
                    enriched_claim['block_index'] = block.index
                    enriched_claim['block_hash'] = block.hash
                    enriched_claim['block_timestamp'] = block.timestamp
                    matching_claims.append(enriched_claim)
                    
        return matching_claims

    def get_studies_by_metadata(self, metadata_filters: Dict[str, Any]) -> List[Dict]:
        """
        Retrieves studies matching the specified metadata filters.
        
        Args:
            metadata_filters: Dictionary of metadata field-value pairs to match
            
        Returns:
            List of matching evidence records
        """
        matching_evidence = []
        
        for block in self.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            
            if not isinstance(evidence, dict) or 'metadata' not in evidence:
                continue
                
            metadata = evidence['metadata']
            
            # Check if all filter criteria match
            if all(key in metadata and metadata[key] == value 
                   for key, value in metadata_filters.items()):
                matching_evidence.append(evidence)
                
        return matching_evidence

    def get_studies_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """
        Retrieves studies containing any of the specified keywords.
        
        Args:
            keywords: List of keywords to match
            
        Returns:
            List of matching evidence records
        """
        matching_evidence = []
        
        for block in self.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            
            if not isinstance(evidence, dict) or 'metadata' not in evidence:
                continue
                
            metadata = evidence['metadata']
            
            # Check if any keyword matches
            if 'keywords' in metadata and any(kw in metadata['keywords'] for kw in keywords):
                matching_evidence.append(evidence)
                
        return matching_evidence

    def is_chain_valid(self) -> bool:
        """
        Validates the integrity of the blockchain.
        
        Returns:
            bool: True if the chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Verify the current block's hash
            if not current_block.verify_hash():
                print(f"Block {i} has an invalid hash!")
                return False
                
            # Verify that current block's previous_hash matches the previous block's hash
            if current_block.previous_hash != previous_block.hash:
                print(f"Block {i} has a mismatched previous hash!")
                return False
                
            # Optional: Verify block signature if implemented
            if hasattr(current_block, 'signature') and current_block.signature:
                if not current_block.verify_signature():
                    print(f"Block {i} has an invalid signature!")
                    return False
                    
        return True

    def search_full_text(self, query: str) -> List[Dict]:
        """
        Simple full-text search across all blocks.
        
        Args:
            query: The search text to look for
            
        Returns:
            List of matching blocks with relevance scores
        """
        query = query.lower()
        results = []
        
        for block in self.chain[1:]:  # Skip genesis block
            # Convert block data to string for searching
            block_str = json.dumps(block.evidence).lower()
            
            # Simple relevance scoring based on term frequency
            relevance = block_str.count(query)
            
            if relevance > 0:
                results.append({
                    'block_index': block.index,
                    'block_hash': block.hash,
                    'relevance': relevance,
                    'evidence_id': block.evidence.get('evidence_id'),
                    'title': block.evidence.get('title', ''),
                    'timestamp': block.timestamp
                })
                
        # Sort by relevance (highest first)
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return results

    def add_evidence_batch(self, evidence_list: List[StudyEvidence]) -> List[str]:
        """
        Adds multiple evidence records as a batch.
        
        Args:
            evidence_list: List of evidence records to add
            
        Returns:
            List of hash values for the added blocks
        """
        block_hashes = []
        
        for evidence in evidence_list:
            block_hash = self.add_block(evidence)
            block_hashes.append(block_hash)
            
        return block_hashes

    def get_blockchain_stats(self) -> Dict[str, Any]:
        """
        Provides statistics about the blockchain.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_blocks': len(self.chain),
            'evidence_blocks': len(self.chain) - 1,  # Exclude genesis
            'unique_studies': len(self.study_index),
            'unique_claims': len(self.claim_index),
            'unique_components': len(self.component_index),
            'last_block_time': datetime.fromtimestamp(self.last_block().timestamp).isoformat(),
            'average_block_size': sum(len(json.dumps(block.evidence)) for block in self.chain) / len(self.chain),
            'chain_valid': self.is_chain_valid()
        }

    def save_blockchain(self, filename: str) -> None:
        """
        Saves the entire blockchain to a JSON file.
        
        Args:
            filename: Path to save the blockchain
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        chain_data = {
            'chain': [block.__dict__ for block in self.chain],
            'indices': {
                'block_index': self.block_index,
                'claim_index': {k: list(v) for k, v in self.claim_index.items()},
                'component_index': {k: list(v) for k, v in self.component_index.items()},
                'study_index': {k: list(v) for k, v in self.study_index.items()}
            },
            'metadata': {
                'version': '2.0',
                'timestamp': datetime.now().isoformat(),
                'block_count': len(self.chain)
            }
        }

        with open(filename, 'w') as f:
            json.dump(chain_data, f, indent=2)

    @staticmethod
    def load_blockchain(filename: str) -> 'Blockchain':
        """
        Loads a blockchain from a JSON file.
        
        Args:
            filename: Path to the blockchain file
            
        Returns:
            Reconstructed Blockchain object
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        blockchain = Blockchain()
        
        # Clear the default genesis block
        blockchain.chain = []
        
        # Reconstruct blocks
        for block_dict in data['chain']:
            blockchain.chain.append(Block.from_dict(block_dict))
            
        # Reconstruct indices
        indices = data.get('indices', {})
        
        blockchain.block_index = indices.get('block_index', {})
        
        # Convert lists back to sets for the other indices
        blockchain.claim_index = {k: set(v) for k, v in indices.get('claim_index', {}).items()}
        blockchain.component_index = {k: set(v) for k, v in indices.get('component_index', {}).items()}
        blockchain.study_index = {k: set(v) for k, v in indices.get('study_index', {}).items()}
        
        return blockchain

    @staticmethod
    def create_evidence_from_study(study, evidence_id: str = None) -> StudyEvidence:
        """
        Creates a StudyEvidence object from a Study object.
        
        Args:
            study: The Study object to convert
            evidence_id: Optional ID for the evidence (generates one if not provided)
            
        Returns:
            StudyEvidence object
        """
        if not evidence_id:
            evidence_id = str(uuid.uuid4())
            
        # Extract metadata from the study
        metadata = StudyMetadata(
            study_id=study.id,
            title=study.name,
            authors=[],  # Would need to extract from study
            institutions=[],  # Would need to extract from study
            funding_sources=[],
            publication_status="unpublished",
            keywords=[]
        )
        
        # Extract evidence claims if they exist
        evidence_claims = []
        if hasattr(study, 'evidence_claims') and study.evidence_claims:
            for claim_data in study.evidence_claims:
                claim = EvidenceClaim(
                    claim_id=claim_data.get('id', str(uuid.uuid4())),
                    claim_text=claim_data.get('text', ''),
                    claim_type=claim_data.get('type', 'associative'),
                    evidence_type=claim_data.get('evidence_type', 'statistical'),
                    source_id=study.id,
                    source_type=claim_data.get('source_type', 'primary_study'),
                    confidence_score=claim_data.get('confidence', 0.5),
                    supporting_data=claim_data.get('supporting_data', {}),
                    metadata=claim_data.get('metadata', {}),
                    created_at=claim_data.get('created_at', datetime.now().isoformat()),
                    created_by=claim_data.get('created_by', 'system')
                )
                evidence_claims.append(claim)
        
        # Extract study components
        study_components = []
        
        # Create the StudyEvidence object
        evidence = StudyEvidence(
            evidence_id=evidence_id,
            study_id=study.id,
            title=study.name,
            description=getattr(study.study_design, 'description', ''),
            intervention_text='',  # Would need to extract from study
            outcome_results={},  # Would need to extract from study
            evidence_claims=evidence_claims,
            study_components=study_components,
            metadata=metadata
        )
        
        return evidence

    def add_study_evidence(self, study, creator_id: str = None) -> str:
        """
        Converts a Study object to evidence and adds it to the blockchain.
        
        Args:
            study: The Study object to add
            creator_id: ID of the creator
            
        Returns:
            Hash of the created block
        """
        evidence = self.create_evidence_from_study(study)
        return self.add_block(evidence, creator_id=creator_id)

    def export_study_evidence(self, study_id: str, format: str = "json") -> Dict[str, Any]:
        """
        Exports all evidence for a study in the specified format.
        
        Args:
            study_id: ID of the study to export
            format: Export format ("json" or "csv")
            
        Returns:
            Dictionary with export data
        """
        if study_id not in self.study_index:
            return {"error": f"Study with ID {study_id} not found"}
            
        study_blocks = [self.chain[idx] for idx in self.study_index[study_id]]
        
        # Compile evidence data
        evidence_data = {
            "study_id": study_id,
            "export_date": datetime.now().isoformat(),
            "format": format,
            "evidence_count": len(study_blocks),
            "evidences": [block.evidence for block in study_blocks]
        }
        
        return evidence_data
    
    def get_evidence_by_time_range(self, start_time: float, end_time: float) -> List[Dict]:
        """
        Retrieves evidence submitted within a specific time range.
        
        Args:
            start_time: Start timestamp (UNIX time)
            end_time: End timestamp (UNIX time)
            
        Returns:
            List of evidence records within the time range
        """
        return [
            block.evidence for block in self.chain
            if start_time <= block.timestamp <= end_time
        ]
    
    def get_related_evidence(self, evidence_id: str, max_results: int = 10) -> List[Dict]:
        """
        Finds evidence related to a specific evidence record.
        
        Args:
            evidence_id: ID of the evidence to find related records for
            max_results: Maximum number of results to return
            
        Returns:
            List of related evidence records with relevance scores
        """
        # Get the source evidence
        source_evidence = self.get_evidence_by_id(evidence_id)
        if not source_evidence:
            return []
            
        # Extract keywords and metadata for comparison
        if isinstance(source_evidence, dict):
            keywords = source_evidence.get('metadata', {}).get('keywords', [])
            study_id = source_evidence.get('study_id')
        else:
            keywords = getattr(getattr(source_evidence, 'metadata', {}), 'keywords', [])
            study_id = getattr(source_evidence, 'study_id', None)
            
        # Simple relatedness scoring
        scored_evidence = []
        
        for block in self.chain[1:]:  # Skip genesis block
            # Skip the source evidence itself
            if block.evidence.get('evidence_id') == evidence_id:
                continue
                
            # Calculate relatedness score
            score = 0
            target_evidence = block.evidence
            
            # Check for same study
            if target_evidence.get('study_id') == study_id:
                score += 5
                
            # Check for keyword overlap
            target_keywords = target_evidence.get('metadata', {}).get('keywords', [])
            if isinstance(target_keywords, list):
                common_keywords = set(keywords) & set(target_keywords)
                score += len(common_keywords) * 2
                
            # Only include if there's some relation
            if score > 0:
                scored_evidence.append({
                    'evidence': target_evidence,
                    'relatedness_score': score,
                    'block_index': block.index,
                    'block_hash': block.hash
                })
                
        # Sort by relevance and limit results
        scored_evidence.sort(key=lambda x: x['relatedness_score'], reverse=True)
        return scored_evidence[:max_results]

    def create_evidence_snapshot(self, snapshot_id: str = None) -> Dict[str, Any]:
        """
        Creates a verified snapshot of the current blockchain state.
        
        Args:
            snapshot_id: Optional ID for the snapshot
            
        Returns:
            Snapshot data with verification information
        """
        if not snapshot_id:
            snapshot_id = str(uuid.uuid4())
            
        # Calculate a merkle root or similar verification hash
        verification_hash = hashlib.sha256(
            json.dumps([block.hash for block in self.chain], sort_keys=True).encode()
        ).hexdigest()
        
        snapshot = {
            'snapshot_id': snapshot_id,
            'timestamp': datetime.now().isoformat(),
            'chain_length': len(self.chain),
            'last_block_hash': self.last_block().hash,
            'verification_hash': verification_hash,
            'block_count': len(self.chain),
            'study_count': len(self.study_index),
            'claim_count': len(self.claim_index)
        }
        
        return snapshot

    def add_claim_to_evidence(self, evidence_id: str, claim: EvidenceClaim) -> bool:
        """
        Adds a new claim to existing evidence and creates a new block.
        
        Args:
            evidence_id: ID of the evidence to update
            claim: The new claim to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Find the source evidence
        source_evidence = self.get_evidence_by_id(evidence_id)
        if not source_evidence:
            return False
            
        # Create a copy of the evidence
        if isinstance(source_evidence, dict):
            updated_evidence = source_evidence.copy()
            
            # Ensure evidence_claims exists and is a list
            if 'evidence_claims' not in updated_evidence:
                updated_evidence['evidence_claims'] = []
                
            # Add the new claim
            updated_evidence['evidence_claims'].append(claim.__dict__)
            
            # Update last_updated
            updated_evidence['last_updated'] = datetime.now().isoformat()
            
            # Create a new evidence object
            new_evidence = StudyEvidence.__new__(StudyEvidence)
            for key, value in updated_evidence.items():
                setattr(new_evidence, key, value)
                
            # Add a new block with the updated evidence
            self.add_block(new_evidence)
            return True
            
        return False
        
    @staticmethod
    def verify_blockchain_file(filename: str) -> Dict[str, Any]:
        """
        Verifies the integrity of a blockchain file without loading it fully.
        
        Args:
            filename: Path to the blockchain file
            
        Returns:
            Verification results
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            chain_data = data.get('chain', [])
            
            # Basic validation
            if not chain_data or not isinstance(chain_data, list):
                return {
                    'valid': False,
                    'error': 'Invalid blockchain format: missing or invalid chain data'
                }
                
            # Check genesis block
            if len(chain_data) == 0:
                return {
                    'valid': False,
                    'error': 'Invalid blockchain: no blocks found'
                }
                
            genesis = chain_data[0]
            if genesis.get('previous_hash') != '0' or genesis.get('index') != 0:
                return {
                    'valid': False,
                    'error': 'Invalid blockchain: incorrect genesis block'
                }
                
            # Verify block connections
            for i in range(1, len(chain_data)):
                current = chain_data[i]
                previous = chain_data[i-1]
                
                if current.get('previous_hash') != previous.get('hash'):
                    return {
                        'valid': False,
                        'error': f'Invalid blockchain: hash mismatch at block {i}'
                    }
                    
                # Verify each block's hash (optional, could be expensive for large chains)
                if i < min(10, len(chain_data)):  # Only check first few blocks
                    block = Block.from_dict(current)
                    if block.hash != block.calculate_hash():
                        return {
                            'valid': False,
                            'error': f'Invalid blockchain: incorrect hash at block {i}'
                        }
            
            return {
                'valid': True,
                'block_count': len(chain_data),
                'first_block_time': datetime.fromtimestamp(chain_data[0].get('timestamp', 0)).isoformat(),
                'last_block_time': datetime.fromtimestamp(chain_data[-1].get('timestamp', 0)).isoformat()
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Verification failed: {str(e)}'
            }

    def get_evidence_provenance(self, evidence_id: str) -> List[Dict]:
        """
        Tracks the provenance (history) of evidence through the blockchain.
        
        Args:
            evidence_id: ID of the evidence to track
            
        Returns:
            List of evidence versions in chronological order
        """
        provenance = []
        study_id = None
        
        # Find the initial evidence to get the study_id
        initial_evidence = self.get_evidence_by_id(evidence_id)
        if not initial_evidence:
            return []
            
        if isinstance(initial_evidence, dict):
            study_id = initial_evidence.get('study_id')
        else:
            study_id = getattr(initial_evidence, 'study_id', None)
            
        if not study_id:
            return []
            
        # Get all blocks for this study
        study_blocks = self.get_blocks_by_study_id(study_id)
        
        # Sort chronologically
        study_blocks.sort(key=lambda block: block.timestamp)
        
        # Build the provenance chain
        for block in study_blocks:
            evidence = block.evidence
            
            provenance.append({
                'block_index': block.index,
                'block_hash': block.hash,
                'timestamp': block.timestamp,
                'iso_time': datetime.fromtimestamp(block.timestamp).isoformat(),
                'evidence_id': evidence.get('evidence_id'),
                'version': len(provenance) + 1,
                'creator_id': block.creator_id,
                'claim_count': len(evidence.get('evidence_claims', [])),
                'component_count': len(evidence.get('study_components', []))
            })
            
        return provenance

    def get_studies_by_protocol_section_title(self, section_title: str) -> List[Block]:
        """
        Retrieves studies with a protocol section matching the specified title.
        
        Args:
            section_title: Title of the protocol section to match
            
        Returns:
            List of matching blocks
        """
        matching_blocks = []
        
        for block in self.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            
            if not isinstance(evidence, dict) or 'study_components' not in evidence:
                continue
                
            # Look for protocol components
            for component in evidence['study_components']:
                if not isinstance(component, dict):
                    continue
                    
                if component.get('component_type') == 'study_protocol':
                    if 'details' in component and 'sections' in component['details']:
                        # Check each section for a title match
                        for section in component['details']['sections']:
                            if isinstance(section, dict) and section.get('title', '').lower() == section_title.lower():
                                matching_blocks.append(block)
                                break
                
        return matching_blocks

    def get_studies_by_statistical_test_name(self, test_name: str) -> List[Block]:
        """
        Retrieves studies that used a specific statistical test.
        
        Args:
            test_name: Name of the statistical test to match
            
        Returns:
            List of matching blocks
        """
        matching_blocks = []
        
        for block in self.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            
            if not isinstance(evidence, dict) or 'study_components' not in evidence:
                continue
                
            # Look for analysis components that might contain statistical tests
            for component in evidence['study_components']:
                if not isinstance(component, dict):
                    continue
                    
                if component.get('component_type') == 'study_analysis':
                    if 'details' in component and 'statistical_tests' in component['details']:
                        # Check each test for a name match
                        for test in component['details']['statistical_tests']:
                            if isinstance(test, dict) and test.get('test_name', '').lower() == test_name.lower():
                                matching_blocks.append(block)
                                break
                
        return matching_blocks

    def add_sample_block(self):
        """
        Adds a sample block with realistic evidence data to the blockchain.
        Primarily used for demonstration and testing.
        
        Returns:
            self: The blockchain instance with the sample block added
        """
        # Create sample metadata
        metadata = {
            "study_id": f"study_{len(self.chain)}",
            "title": f"Sample Study {len(self.chain)}",
            "authors": ["Researcher A", "Researcher B"],
            "institutions": ["University Research Center"],
            "funding_sources": [{"name": "Research Foundation", "grant_id": "RF-2023-001"}],
            "publication_status": "preprint",
            "population_indicators": {
                "age_group": "adults",
                "health_status": "general",
                "region": "global"
            },
            "keywords": ["intervention", "sample", "testing"]
        }
        
        # Create a sample claim
        claim = {
            "claim_id": f"claim_{len(self.chain)}",
            "claim_text": "The intervention shows statistically significant improvement compared to control",
            "claim_type": "causal",
            "evidence_type": "experimental",
            "source_id": f"source_{len(self.chain)}",
            "source_type": "primary_study",
            "confidence_score": 0.85,
            "supporting_data": {
                "p_value": 0.012,
                "effect_size": 0.42,
                "sample_size": 120
            },
            "metadata": {
                "population": "adults aged 18-65",
                "condition": "general health"
            },
            "created_at": datetime.now().isoformat(),
            "created_by": "system"
        }
        
        # Create a sample protocol section
        protocol_section = {
            "section_id": f"protocol_section_{len(self.chain)}",
            "title": "Methodology",
            "body": "This study used a randomized controlled design with double blinding.",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "author": "Researcher A",
            "contributors": ["Researcher B"],
            "contributions": {},
            "revision_history": []
        }
        
        # Create protocol component
        protocol_component = {
            "component_id": f"protocol_{len(self.chain)}",
            "component_type": "study_protocol",
            "details": {
                "sections": [protocol_section],
                "metadata": {
                    "version": "1.0",
                    "approved_date": datetime.now().isoformat()
                },
                "version_history": []
            },
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        # Create a sample data source
        data_source = {
            "source_id": f"datasource_{len(self.chain)}",
            "name": "Trial Dataset A",
            "data_source_type": "csv",
            "source_details": {"filename": "trial_data.csv"},
            "rows": 120,
            "columns": 15,
            "column_names": ["patient_id", "age", "treatment_group", "outcome"],
            "column_metadata": {},
            "column_types": {
                "patient_id": "categorical",
                "age": "continuous",
                "treatment_group": "categorical",
                "outcome": "continuous"
            },
            "privacy_level": "standard",
            "access_restrictions": [],
            "data_provenance": {},
            "created_at": datetime.now().isoformat()
        }
        
        # Create a sample statistical test
        statistical_test = {
            "test_id": f"test_{len(self.chain)}",
            "test_name": "Two-sample t-test",
            "test_type": "parametric",
            "variables": {
                "outcome": ["outcome"],
                "predictor": ["treatment_group"]
            },
            "parameters": {
                "alpha": 0.05,
                "tails": 2
            },
            "results": {
                "t_statistic": 2.58,
                "p_value": 0.012,
                "mean_diff": 1.85,
                "conf_int_lower": 0.42,
                "conf_int_upper": 3.28
            },
            "assumptions_checked": {
                "normality": True,
                "homogeneity": True
            },
            "analysis_details": "Compared mean outcome between treatment and control groups",
            "software_used": {"name": "R", "version": "4.1.2"},
            "created_at": datetime.now().isoformat(),
            "executed_by": "Researcher A"
        }
        
        # Create analysis component
        analysis_component = {
            "component_id": f"analysis_{len(self.chain)}",
            "component_type": "study_analysis",
            "details": {
                "data_sources": [data_source],
                "intervention_variables": ["treatment_group"],
                "outcome_variables": ["outcome"],
                "confounding_variables": ["age"],
                "statistical_tests": [statistical_test],
                "analysis_plan": {},
                "preregistered": True,
                "metadata": {}
            },
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        # Create a sample literature review section
        lit_review_section = {
            "title": "Previous Studies",
            "body": "Several previous studies have investigated similar interventions with mixed results.",
            "citations": [
                {"id": "citation_1", "text": "Smith et al. (2020)", "doi": "10.1000/xyz123"}
            ]
        }
        
        # Create literature review component
        lit_review_component = {
            "component_id": f"litreview_{len(self.chain)}",
            "component_type": "literature_review",
            "details": {
                "thematic_sections": [lit_review_section],
                "metadata": {
                    "databases_searched": ["PubMed", "Scopus"],
                    "date_range": "2010-2023"
                }
            },
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        # Create evidence
        evidence = {
            "evidence_id": f"evidence_{len(self.chain)}",
            "study_id": metadata["study_id"],
            "title": metadata["title"],
            "description": "A randomized controlled trial investigating the effects of the intervention.",
            "intervention_text": "Participants received the intervention for 12 weeks under controlled conditions.",
            "outcome_results": {
                "primary": {
                    "outcome": {
                        "value": 1.85,
                        "p_value": 0.012,
                        "significant": True
                    }
                }
            },
            "evidence_claims": [claim],
            "study_components": [protocol_component, analysis_component, lit_review_component],
            "metadata": metadata,
            "evidence_quality": {
                "bias_risk": "low",
                "methodology_score": 8.5
            },
            "certainty_rating": "moderate",
            "external_validations": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add the block
        self.add_block(evidence, creator_id="system")
        
        return self

# # Example usage of the enhanced blockchain
# if __name__ == '__main__':
#     # Initialize blockchain
#     blockchain = Blockchain()
#     print(f"Initialized blockchain with genesis block: {blockchain.last_block().hash}")
    
#     # Create sample metadata
#     metadata = StudyMetadata(
#         study_id="study123",
#         title="Effect of Intervention X on Outcome Y",
#         authors=["Researcher A", "Researcher B"],
#         institutions=["University 1", "Research Center 2"],
#         funding_sources=[{"name": "Health Foundation", "grant_id": "HF-2023-456"}],
#         publication_status="preprint",
#         keywords=["intervention x", "outcome y", "randomized trial"]
#     )
    
#     # Create a sample statistical test
#     test1 = StatisticalTest(
#         test_id="test123",
#         test_name="Two-sample t-test",
#         test_type="parametric",
#         variables={
#             "outcome": ["blood_pressure"],
#             "predictor": ["treatment_group"]
#         },
#         parameters={
#             "alpha": 0.05,
#             "tails": 2
#         },
#         results={
#             "t_statistic": -3.5,
#             "p_value": 0.002,
#             "mean_diff": -10.2,
#             "conf_int_lower": -15.3,
#             "conf_int_upper": -5.1
#         },
#         assumptions_checked={
#             "normality": True,
#             "homogeneity": True
#         },
#         analysis_details="Compared mean blood pressure between treatment and control groups",
#         software_used={"name": "R", "version": "4.1.2", "packages": ["stats"]},
#         created_at=datetime.now().isoformat(),
#         executed_by="researcher_a"
#     )
    
#     # Create a sample data source
#     data_source = DataSource(
#         source_id="datasource123",
#         name="Clinical Trial Data",
#         data_source_type="csv",
#         source_details={"filename": "trial_data.csv"},
#         rows=120,
#         columns=15,
#         column_names=["patient_id", "age", "sex", "treatment_group", "blood_pressure"],
#         column_metadata={
#             "blood_pressure": {"unit": "mmHg", "measurement_time": "baseline"}
#         },
#         column_types={
#             "patient_id": "categorical",
#             "age": "continuous",
#             "sex": "categorical",
#             "treatment_group": "categorical",
#             "blood_pressure": "continuous"
#         }
#     )
    
#     # Create a sample study analysis component
#     study_analysis = StudyAnalysis(
#         component_id="analysis123",
#         data_sources=[data_source],
#         intervention_variables=["treatment_group"],
#         outcome_variables=["blood_pressure"],
#         confounding_variables=["age", "sex"],
#         statistical_tests=[test1],
#         preregistered=True
#     )
    
#     # Create a sample claim
#     claim = EvidenceClaim(
#         claim_id="claim123",
#         claim_text="Intervention X significantly reduces blood pressure compared to placebo",
#         claim_type="causal",
#         evidence_type="experimental",
#         source_id="study123",
#         source_type="primary_study",
#         confidence_score=0.85,
#         supporting_data={
#             "effect_size": "moderate",
#             "p_value": 0.002,
#             "sample_size": 120
#         },
#         metadata={
#             "replicated": False,
#             "population": "adults with hypertension"
#         },
#         created_at=datetime.now().isoformat(),
#         created_by="researcher_a"
#     )
    
#     # Create sample evidence
#     evidence = StudyEvidence(
#         evidence_id="evidence123",
#         study_id="study123",
#         title="Effect of Intervention X on Blood Pressure",
#         description="A randomized controlled trial evaluating the efficacy of Intervention X",
#         intervention_text="Daily administration of Intervention X (50mg) for 12 weeks",
#         outcome_results={
#             "primary": {
#                 "blood_pressure_reduction": {
#                     "value": -10.2,
#                     "unit": "mmHg",
#                     "p_value": 0.002,
#                     "significant": True
#                 }
#             }
#         },
#         evidence_claims=[claim],
#         study_components=[study_analysis],
#         metadata=metadata
#     )
    
#     # Add evidence to blockchain
#     block_hash = blockchain.add_block(evidence, creator_id="researcher_a")
#     print(f"Added evidence block with hash: {block_hash}")
    
#     # Verify the blockchain
#     is_valid = blockchain.is_chain_valid()
#     print(f"Blockchain validity: {is_valid}")
    
#     # Save to file
#     filename = "evidence_blockchain.json"
#     blockchain.save_blockchain(filename)
#     print(f"Blockchain saved to {filename}")
    
#     # Get blockchain statistics
#     stats = blockchain.get_blockchain_stats()
#     print("Blockchain statistics:")
#     for key, value in stats.items():
#         print(f"  {key}: {value}")
    
#     # Search for evidence
#     search_results = blockchain.search_full_text("blood pressure")
#     print(f"\nFound {len(search_results)} results for 'blood pressure':")
#     for result in search_results:
#         print(f"  Block {result['block_index']}: {result['title']} (relevance: {result['relevance']})")