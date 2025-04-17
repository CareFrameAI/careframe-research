# # Dependencies for BioNLP module
# # NOTE: There are dependency conflicts between spacy/scispacy and other packages like openai and anthropic
# # It's recommended to create a separate conda/virtual environment for this module
# #
# # Environment setup:
# # 1. Create a new environment: conda create -n bionlp python=3.9
# # 2. Activate: conda activate bionlp
# # 3. Install the following packages:
# 
# # Install spacy and scispacy with compatible typing-extensions
# # pip install "typing-extensions<4.6.0,>=3.7.4.1"
# # pip install "pydantic<2.0.0,>=1.8.2"
# # pip install scispacy==0.5.3
# # pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
# 
# # Install entity linkers for different ontologies
# # pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz
# 
# # Install additional packages
# # pip install icd10 pyspellchecker fastapi uvicorn
# # Install scispacy and its models
# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz

# # Install additional scispacy models for entity linking
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bionlp13cg_md-0.5.0.tar.gz

# # Install required linkers
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_scibert-0.5.0.tar.gz

# # Install ICD-10 package
# pip install icd10

# # Install spellchecker
# pip install pyspellchecker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import spacy
import json
import logging

import os
import spacy
from spacy import displacy
from typing import List, Dict, Any
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from collections import defaultdict
import logging
import re
import json
import icd10
import csv
from spellchecker import spellchecker

class MedicalTextReasoner:
    def __init__(self, umls_api_key: str, loinc_csv_path: str, loinc_to_hpo_csv_path: str, threshold: float = 0.7):
        """
        Initialize the MedicalTextReasoner with enhanced configuration.

        Args:
            umls_api_key (str): UMLS license key
            loinc_csv_path (str): Path to Loinc.csv file
            loinc_to_hpo_csv_path (str): Path to loinc_to_hpo_mapping.csv file
            threshold (float): Confidence threshold for entity linking
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            os.environ['UMLS_LICENSE_KEY'] = umls_api_key
            self.threshold = threshold
            self.loinc_csv_path = loinc_csv_path
            self.loinc_to_hpo_csv_path = loinc_to_hpo_csv_path
            self._initialize_nlp_pipelines()
            self._load_loinc_details()
            self._load_loinc_mapping()
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_nlp_pipelines(self) -> None:
        """Initialize and configure NLP pipelines."""
        try:
            # Load the large scientific model
            self.nlp_umls = spacy.load("en_core_sci_lg")
            self.nlp_mesh = spacy.load("en_core_sci_lg")
            self.nlp_rxnorm = spacy.load("en_core_sci_lg")
            self.nlp_go = spacy.load("en_core_sci_lg")
            self.nlp_hpo = spacy.load("en_core_sci_lg")
            
            # Add AbbreviationDetector and Sentencizer
            for nlp in [self.nlp_umls, self.nlp_mesh, self.nlp_rxnorm, self.nlp_go, self.nlp_hpo]:
                if "abbreviation_detector" not in nlp.pipe_names:
                    nlp.add_pipe("abbreviation_detector", before="ner")
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
            
            # Add scispacy linkers to each pipeline
            self.nlp_umls.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": "umls",
                    "threshold": self.threshold,
                    "max_entities_per_mention": 3,
                },
                last=True
            )
            self.nlp_mesh.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": "mesh",
                    "threshold": self.threshold,
                    "max_entities_per_mention": 3,
                },
                last=True
            )
            self.nlp_rxnorm.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": "rxnorm",
                    "threshold": self.threshold,
                    "max_entities_per_mention": 3,
                },
                last=True
            )
            self.nlp_go.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": "go",
                    "threshold": self.threshold,
                    "max_entities_per_mention": 3,
                },
                last=True
            )
            self.nlp_hpo.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": "hpo",
                    "threshold": self.threshold,
                    "max_entities_per_mention": 3,
                },
                last=True
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP pipelines: {str(e)}")
            raise

    def _load_loinc_mapping(self) -> None:
        """
        Load LOINC to HPO mapping from loinc_to_hpo_mapping.csv.
        """
        self.loinc_to_hpo = defaultdict(set)
        try:
            with open(self.loinc_to_hpo_csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    loinc_num = row['LOINC_NUM'].strip()
                    hpo_id = row['HPO_ID'].strip()
                    self.loinc_to_hpo[loinc_num].add(hpo_id)
            self.logger.info(f"Loaded LOINC to HPO mappings for {len(self.loinc_to_hpo)} LOINC codes.")
        except Exception as e:
            self.logger.error(f"Failed to load LOINC to HPO mapping: {str(e)}")
            raise

    def _load_loinc_details(self) -> None:
        """
        Load LOINC details from Loinc.csv to map LOINC_NUM to LONG_COMMON_NAME.
        """
        self.loinc_details = {}
        try:
            with open(self.loinc_csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    loinc_num = row['LOINC_NUM'].strip()
                    long_common_name = row['LONG_COMMON_NAME'].strip().lower()
                    self.loinc_details[loinc_num] = long_common_name
            self.logger.info(f"Loaded details for {len(self.loinc_details)} LOINC codes.")
        except Exception as e:
            self.logger.error(f"Failed to load LOINC details: {str(e)}")
            raise

    def extract_concepts(self, doc, linker, linker_name: str) -> List[Dict[str, Any]]:
        """
        Extract medical concepts from the processed document.

        Args:
            doc: Processed SpaCy Doc object
            linker: Entity linker from the NLP pipeline
            linker_name (str): Name of the linker ("umls", "mesh", etc.)

        Returns:
            List of extracted concepts with detailed information
        """
        concepts = []
        try:
            for entity in doc.ents:
                if entity._.kb_ents:
                    for kb_ent in entity._.kb_ents:
                        cui = kb_ent[0]
                        score = kb_ent[1]
                        concept = linker.kb.cui_to_entity.get(cui)
                        
                        if concept:
                            concept_info = {
                                "entity_text": entity.text,
                                "start_char": entity.start_char,
                                "end_char": entity.end_char,
                                "cui": cui,
                                "score": float(score),  # Convert to float for JSON serialization
                                "canonical_name": concept.canonical_name,
                                "definition": getattr(concept, 'definition', 'N/A'),  # Some KBs might not have definitions
                                "types": list(concept.types),  # Convert set to list
                                "linker_name": linker_name,
                                "label": entity.label_,
                                "context": self._get_entity_context(doc, entity)
                            }
                            
                            # Add semantic type information if available
                            if hasattr(concept, 'semantic_type'):
                                concept_info["semantic_type"] = concept.semantic_type

                            # Attempt to extract ICD10CM codes from UMLS concepts
                            if linker_name == 'umls':
                                icd10cm_codes = [alias for alias in concept.aliases if re.match(r'^[A-TV-Z][0-9]{2}(?:\.[0-9]+)?$', alias)]
                                if icd10cm_codes:
                                    concept_info["icd10cm_codes"] = icd10cm_codes

                            concepts.append(concept_info)
        except Exception as e:
            self.logger.error(f"Error extracting concepts: {str(e)}")
                
        return concepts

    def _get_entity_context(self, doc, entity, window: int = 3) -> str:
        """Get the surrounding context of an entity."""
        start = max(0, entity.start - window)
        end = min(len(doc), entity.end + window)
        return doc[start:end].text

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process medical text with enhanced features.

        Args:
            text (str): Input medical text

        Returns:
            Dict containing processed information
        """
        # Basic text preprocessing
        text = self._preprocess_text(text)
        
        # Process the text with each pipeline
        doc_umls = self.nlp_umls(text)
        doc_mesh = self.nlp_mesh(text)
        doc_rxnorm = self.nlp_rxnorm(text)
        doc_go = self.nlp_go(text)
        doc_hpo = self.nlp_hpo(text)
        
        # Extract concepts for each linker
        concepts = {}
        concepts['umls_concepts'] = self.extract_concepts(doc_umls, self.nlp_umls.get_pipe("scispacy_linker"), 'umls')
        concepts['mesh_concepts'] = self.extract_concepts(doc_mesh, self.nlp_mesh.get_pipe("scispacy_linker"), 'mesh')
        concepts['rxnorm_concepts'] = self.extract_concepts(doc_rxnorm, self.nlp_rxnorm.get_pipe("scispacy_linker"), 'rxnorm')
        concepts['go_concepts'] = self.extract_concepts(doc_go, self.nlp_go.get_pipe("scispacy_linker"), 'go')
        concepts['hpo_concepts'] = self.extract_concepts(doc_hpo, self.nlp_hpo.get_pipe("scispacy_linker"), 'hpo')
        
        # Use one of the docs for sentence splitting, abbreviation detection, etc.
        doc = doc_umls
        
        # Extract ICD-10 information
        icd10_info = self.extract_icd10_info(text)
        
        # Map Labs to LOINC and HPO
        lab_mappings = self._map_labs_to_codes(text)
        
        return {
            **concepts,
            "abbreviations": self.get_abbreviations(doc),
            "sentences": self.extract_sentences(doc),
            "named_entities": self.extract_named_entities(doc),
            "key_phrases": self.extract_key_phrases(doc),
            "negated_concepts": self.extract_negated_concepts(doc),
            "semantic_relations": self.analyze_semantic_relations(doc),
            "term_frequencies": self.get_term_frequency(text),
            "icd10_details": icd10_info,  # Include ICD-10 details in the results
            "lab_mappings": lab_mappings  # Include Lab mappings in the results
        }

    def _map_labs_to_codes(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify lab names in the text and map them to LOINC and HPO codes.

        Args:
            text (str): Input medical text

        Returns:
            List of mappings containing lab name, LOINC_NUM, and associated HPO_IDs
        """
        mappings = []
        try:
            # Tokenize the text
            doc = self.nlp_umls(text)
            
            # Extract potential lab mentions based on LOINC LONG_COMMON_NAME
            for loinc_num, long_common_name in self.loinc_details.items():
                # Use case-insensitive search
                pattern = re.compile(r'\b' + re.escape(long_common_name) + r'\b', re.IGNORECASE)
                matches = pattern.findall(text)
                if matches:
                    for match in matches:
                        hpo_ids = list(self.loinc_to_hpo.get(loinc_num, []))
                        if hpo_ids:
                            mappings.append({
                                "lab_name": match,
                                "LOINC_NUM": loinc_num,
                                "HPO_IDs": hpo_ids
                            })
        except Exception as e:
            self.logger.error(f"Error mapping labs to codes: {str(e)}")
        
        return mappings

    def get_abbreviations(self, doc) -> List[Dict[str, str]]:
        """Extract abbreviations and their expansions."""
        abbreviations = []
        if hasattr(doc._, 'abbreviations'):
            for abrv in doc._.abbreviations:
                abbreviations.append({
                    "short_form": abrv.text,
                    "long_form": abrv._.long_form.text
                })
        return abbreviations

    def _preprocess_text(self, text: str) -> str:
        """Preprocess input text."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.-]', ' ', text)
        return text

    def extract_named_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract named entities with their labels."""
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]

    def extract_key_phrases(self, doc) -> List[Dict[str, Any]]:
        """Extract key phrases based on noun chunks and dependency parsing."""
        key_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in ['nsubj', 'dobj', 'pobj']:
                key_phrases.append({
                    "phrase": chunk.text,
                    "root": chunk.root.text,
                    "dependency": chunk.root.dep_
                })
        return key_phrases

    def extract_negated_concepts(self, doc) -> List[Dict[str, Any]]:
        """Identify negated medical concepts."""
        negation_tokens = {'no', 'not', 'negative', 'deny', 'denies', 'without'}
        negated = []
        
        for ent in doc.ents:
            for token in doc[max(0, ent.start - 3):ent.start]:
                if token.text.lower() in negation_tokens:
                    negated.append({
                        "concept": ent.text,
                        "negation_term": token.text
                    })
        return negated

    def extract_sentences(self, doc) -> List[Dict[str, Any]]:
        """Extract sentences with their medical entities."""
        return [
            {
                "text": sent.text,
                "entities": [
                    {"text": ent.text, "label": ent.label_}
                    for ent in sent.ents
                ]
            }
            for sent in doc.sents
        ]

    def analyze_semantic_relations(self, doc) -> List[Dict[str, Any]]:
        """Analyze semantic relationships between medical concepts."""
        relations = []
        for token in doc:
            if token.dep_ in ["prep", "conj", "compound"]:
                relations.append({
                    "source": token.head.text,
                    "relation": token.dep_,
                    "target": token.text
                })
        return relations

    def get_term_frequency(self, text: str) -> Dict[str, int]:
        """Calculate frequency of medical terms."""
        doc = self.nlp_umls(text)
        term_freq = defaultdict(int)
        for ent in doc.ents:
            term_freq[ent.text.lower()] += 1
        return dict(term_freq)

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts in batch."""
        return [self.process_text(text) for text in texts]

    def export_to_format(self, results: Dict[str, Any], format: str = 'dict') -> Any:
        """Export results in different formats (dict, json, xml)."""
        if format == 'json':
            return json.dumps(results, indent=2)
        elif format == 'xml':
            import xml.etree.ElementTree as ET

            def dict_to_xml(tag, d):
                elem = ET.Element(tag)
                for key, val in d.items():
                    child = ET.SubElement(elem, key)
                    if isinstance(val, dict):
                        child.append(dict_to_xml(key, val))
                    elif isinstance(val, list):
                        for item in val:
                            child.append(dict_to_xml(key[:-1], item))
                    else:
                        child.text = str(val)
                return elem

            root = dict_to_xml('MedicalText', results)
            return ET.tostring(root, encoding='unicode')
        return results

    def save_results(self, results, filename='medical_analysis_results.json'):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def extract_icd10_info(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract ICD-10 codes from text and retrieve their detailed information.

        Args:
            text (str): Input medical text

        Returns:
            List of dictionaries containing ICD-10 code information
        """
        icd10_details = []
        # Regular expression to find ICD-10 codes (e.g., E11.9, I10)
        # Ensure that the dot is preserved by making it part of the capturing group
        icd10_pattern = r'\b([A-TV-Z][0-9]{2}(?:\.[0-9]+)?)\b'
        codes_found = re.findall(icd10_pattern, text.upper())

        unique_codes = set(codes_found)  # Remove duplicates
        for code_str in unique_codes:
            # Standardize the code format to include the dot
            standardized_code = self._standardize_icd10_code(code_str)
            code = icd10.find(standardized_code)
            if code:
                code_info = {
                    "code": code.code,  # This should include the dot
                    "description": code.description,
                    "billable": code.billable,
                    "chapter": code.chapter,
                    "block": code.block,
                    "block_description": code.block_description
                }
                icd10_details.append(code_info)
            else:
                self.logger.warning(f"ICD-10 code {code_str} not found in the icd10 module.")
        
        return icd10_details

    def _standardize_icd10_code(self, code_str: str) -> str:
        """
        Standardize the ICD-10 code to include a decimal point.

        Args:
            code_str (str): Raw ICD-10 code extracted from text

        Returns:
            str: Standardized ICD-10 code with decimal
        """
        # If the code already contains a dot, return as is
        if '.' in code_str:
            return code_str
        # Insert a dot after the third character (e.g., E119 -> E11.9)
        elif len(code_str) > 3:
            return f"{code_str[:3]}.{code_str[3:]}"
        else:
            return code_str  # Return as is if it doesn't match expected format

    def correct_text_after_analysis(self, text: str, analysis_results: Dict[str, Any]) -> str:
        """
        Corrects spelling errors in the text after analysis, including misspelled medical terms.

        Args:
            text (str): Original input medical text.
            analysis_results (Dict[str, Any]): Results from the analysis.

        Returns:
            str: Text with corrected spelling.
        """
        spell = spellchecker.SpellChecker()
        # Build a mapping from entity_text to canonical_name
        medical_terms_map = self._extract_medical_terms_map(analysis_results)
        # Add correct medical terms to the spell checker's dictionary
        spell.word_frequency.load_words(set(medical_terms_map.values()))

        # Process the text and get entities
        doc = self.nlp_umls(text)
        # Build a list of (start_char, end_char, replacement_text)
        replacements = []
        for ent in doc.ents:
            entity_text = ent.text.lower()
            canonical_name = medical_terms_map.get(entity_text, ent.text)
            # Replace if the entity_text is different from the canonical_name
            if entity_text != canonical_name.lower():
                replacements.append((ent.start_char, ent.end_char, canonical_name))
        # Sort replacements by start_char
        replacements.sort(key=lambda x: x[0])

        # Build the corrected text by replacing entities with their canonical names
        corrected_text = ''
        last_idx = 0
        for start_char, end_char, replacement_text in replacements:
            # Append text before the entity
            corrected_text += text[last_idx:start_char]
            # Append the replacement text
            corrected_text += replacement_text
            last_idx = end_char
        # Append any remaining text after the last entity
        corrected_text += text[last_idx:]

        # Now, apply spell checking to the corrected_text
        # Tokenize the corrected_text using spaCy
        corrected_doc = self.nlp_umls(corrected_text)
        final_tokens = []
        for token in corrected_doc:
            if token.is_punct:
                final_tokens.append(token.text)
            elif token.text.lower() in medical_terms_map.values():
                final_tokens.append(token.text_with_ws)
            else:
                # Correct the word using spell checker
                corrected_word = spell.correction(token.text)
                final_tokens.append(corrected_word + token.whitespace_)
        final_corrected_text = ''.join(final_tokens)
        return final_corrected_text

    def _extract_medical_terms_map(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts a mapping from entity_text to canonical_name from the analysis results.

        Args:
            analysis_results (Dict[str, Any]): Results from the analysis.

        Returns:
            Dict[str, str]: A mapping from possibly misspelled medical terms to their correct forms.
        """
        medical_terms_map = {}
        # Extract terms from UMLS concepts
        for concept in analysis_results.get('umls_concepts', []):
            entity_text = concept['entity_text'].lower()
            canonical_name = concept['canonical_name']
            medical_terms_map[entity_text] = canonical_name
        # Similarly, extract terms from other concept types
        for key in ['mesh_concepts', 'rxnorm_concepts', 'hpo_concepts']:
            for concept in analysis_results.get(key, []):
                entity_text = concept['entity_text'].lower()
                canonical_name = concept['canonical_name']
                medical_terms_map[entity_text] = canonical_name
        # Add named entities
        for entity in analysis_results.get('named_entities', []):
            entity_text = entity['text'].lower()
            medical_terms_map[entity_text] = entity['text']
        return medical_terms_map

# Define configuration parameters for initialization
UMLS_API_KEY = ''
LOINC_CSV_PATH = 'Loinc.csv'
LOINC_TO_HPO_CSV_PATH = 'loinc_to_hpo_mapping.csv'
THRESHOLD = 0.7

app = FastAPI()

# Initialize the MedicalTextReasoner instance globally on startup
medical_text_parser = None

@app.on_event("startup")
def startup_event():
    global medical_text_parser
    try:
        # Get UMLS API key from secrets
        from admin.portal import get_secrets_from_database
        secrets = get_secrets_from_database()
        umls_api_key = secrets.get('UMLS_API_KEY', '')
        
        if not umls_api_key:
            logging.warning("UMLS API key not found in secrets database, using empty key")
        else:
            logging.info("UMLS API key retrieved from secrets database")
        
        # Initialize the MedicalTextReasoner instance
        medical_text_parser = MedicalTextReasoner(
            umls_api_key=umls_api_key,
            loinc_csv_path=LOINC_CSV_PATH,
            loinc_to_hpo_csv_path=LOINC_TO_HPO_CSV_PATH,
            threshold=THRESHOLD
        )
        logging.info("MedicalTextReasoner initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing MedicalTextReasoner: {str(e)}")
        raise

class TextRequest(BaseModel):
    text: str

@app.post("/process_text")
async def process_text(request: TextRequest):
    global medical_text_parser
    if not medical_text_parser:
        raise HTTPException(status_code=500, detail="MedicalTextReasoner is not initialized.")

    try:
        result = medical_text_parser.process_text(request.text)
        return result
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing text.")