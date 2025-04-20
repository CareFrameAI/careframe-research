#!/usr/bin/env python3
"""
Privacy Filter Module for PHI/PII Detection and Anonymization
Supports HIPAA (US) and PHIPA (Canada) compliance requirements
Includes malicious content detection and inappropriate language filtering
"""

import re
import json
import os
from typing import Dict, List, Union, Optional, Set, Tuple
import logging

# Import required libraries (assuming they're installed)
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

try:
    import scrubadub
    SCRUBADUB_AVAILABLE = True
    # Make sklearn detectors optional
    try:
        import scrubadub.detectors.sklearn
        SKLEARN_DETECTORS_AVAILABLE = True
    except ImportError:
        SKLEARN_DETECTORS_AVAILABLE = False
except ImportError:
    SCRUBADUB_AVAILABLE = False
    SKLEARN_DETECTORS_AVAILABLE = False

# Import spaCy and profanity filter (required for enhanced detection)
try:
    import spacy
    from spacy.pipeline import EntityRuler
    from spacy.language import Language
    from better_profanity import profanity
    ENHANCED_DETECTION_AVAILABLE = True
    
    @Language.factory("privacy_ruler")
    def create_privacy_ruler(nlp, name):
        return EntityRuler(nlp, overwrite_ents=True)
        
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)

class PrivacyFilter:
    """
    Privacy filter to detect and anonymize PHI/PII in text data according to
    HIPAA (US) and PHIPA (Canada) compliance requirements.
    Also detects malicious content and inappropriate language.
    """
    
    # PHI categories based on HIPAA and PHIPA regulations
    PHI_CATEGORIES = {
        # HIPAA 18 identifiers
        'US': {
            'NAME': 'Person name',
            'AGE': 'Age (>89)',
            'PHONE_NUMBER': 'Phone number',
            'EMAIL_ADDRESS': 'Email address',
            'SSN': 'Social Security Number',
            'MRN': 'Medical Record Number',
            'HEALTH_PLAN_ID': 'Health plan beneficiary number',
            'ACCOUNT_NUMBER': 'Account number',
            'LICENSE_NUMBER': 'Certificate/license number',
            'DEVICE_ID': 'Device identifier',
            'URL': 'Web URL',
            'IP_ADDRESS': 'IP address',
            'BIOMETRIC_ID': 'Biometric identifier',
            'PHOTO': 'Full face photo',
            'ADDRESS': 'Geographic location smaller than state',
            'DATE': 'Dates related to individual',
            'VEHICLE_ID': 'Vehicle identifier',
            'OTHER_ID': 'Any other unique identifying number or code',
            'INAPPROPRIATE_LANGUAGE': 'Inappropriate or abusive language'
        },
        # PHIPA specific identifiers (in addition to HIPAA)
        'CANADA': {
            'OHIP': 'Ontario Health Insurance Plan number',
            'HEALTH_CARD': 'Provincial health card number',
            'SIN': 'Social Insurance Number',
            'DRIVER_LICENSE': 'Driver\'s license number',
            'POSTAL_CODE': 'Postal code'
        }
    }
    
    def __init__(self, 
                 use_presidio: bool = True, 
                 use_scrubadub: bool = True,
                 use_regex: bool = True,
                 regions: List[str] = None,
                 language: str = 'en',
                 patterns_file: str = None):
        """
        Initialize the privacy filter with chosen libraries and configuration.
        
        Args:
            use_presidio: Whether to use Microsoft Presidio for detection
            use_scrubadub: Whether to use scrubadub for detection
            use_regex: Whether to use custom regex patterns for detection
            regions: List of regions to apply (e.g., ['US', 'CANADA'])
            language: Language of the text to analyze
            patterns_file: Path to a custom patterns.json file
        """
        self.regions = regions or ['US', 'CANADA']
        self.language = language
        self.use_regex = use_regex
        
        # Initialize available libraries
        self.presidio_analyzer = None
        self.presidio_anonymizer = None
        self.scrubber = None
        
        # Load patterns for enhanced detection
        self.patterns_file = patterns_file or os.path.join(os.path.dirname(__file__), 'patterns.json')
        self.patterns = []
        self.malicious_patterns = []
        self.load_patterns()
        
        # Initialize spaCy for enhanced detection
        self.nlp = None
        self.entity_ruler = None
        if ENHANCED_DETECTION_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                # Initialize EntityRuler
                if "privacy_ruler" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("privacy_ruler", before='ner')
                self.entity_ruler = self.nlp.get_pipe("privacy_ruler")
                self.add_patterns_to_entity_ruler()
                
                # Initialize profanity filter
                self.profanity = profanity
            except Exception as e:
                logger.warning(f"Failed to initialize spaCy: {e}")
        
        if use_presidio and PRESIDIO_AVAILABLE:
            # Set up NLP engine with English defaults
            nlp_engine = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": language, "model_name": "en_core_web_lg"}]}).create_engine()
            
            # Create analyzer with the NLP engine
            self.presidio_analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            
            # Create the anonymizer engine
            self.presidio_anonymizer = AnonymizerEngine()
        elif use_presidio and not PRESIDIO_AVAILABLE:
            logger.warning("Microsoft Presidio requested but not installed. Install with: pip install presidio-analyzer presidio-anonymizer spacy && python -m spacy download en_core_web_lg")
            
        if use_scrubadub and SCRUBADUB_AVAILABLE:
            # Configure scrubadub with appropriate detectors
            self.scrubber = scrubadub.Scrubber()
            
            # Add detectors specifically for healthcare PHI
            if SKLEARN_DETECTORS_AVAILABLE:
                self.scrubber.add_detector(scrubadub.detectors.sklearn.SklearnDetailedDetector())
                
            # For each country's detectors
            if 'US' in self.regions:
                # Add US specific detectors
                try:
                    self.scrubber.add_detector(scrubadub.detectors.SSNDetector)
                except:
                    pass
            
            if 'CANADA' in self.regions:
                # Add Canada specific detectors
                try:
                    self.scrubber.add_detector(scrubadub.detectors.SINDetector)
                except:
                    pass
                
        elif use_scrubadub and not SCRUBADUB_AVAILABLE:
            logger.warning("Scrubadub requested but not installed. Install with: pip install scrubadub")
            if not SKLEARN_DETECTORS_AVAILABLE:
                logger.warning("For enhanced detection capabilities, install sklearn extension: pip install scrubadub[sklearn]")
                
    def load_patterns(self):
        """Load patterns from the JSON file for enhanced detection"""
        try:
            with open(self.patterns_file, 'r') as f:
                data = json.load(f)
                self.patterns = data.get('patterns', [])
                self.malicious_patterns = data.get('malicious_patterns', [])
        except Exception as e:
            logger.warning(f"Could not load patterns from {self.patterns_file}: {e}")
            # Initialize with empty patterns
            self.patterns = []
            self.malicious_patterns = []
    
    def add_patterns_to_entity_ruler(self):
        """Add patterns to the spaCy entity ruler"""
        if not self.entity_ruler:
            return
            
        ruler_patterns = []
        for pattern in self.patterns:
            ruler_patterns.append({
                'label': pattern.get('category', 'PHI'),
                'pattern': pattern.get('replacement', '[UNKNOWN]'),
                'id': pattern.get('name', 'unknown')
            })
        
        # Only add patterns if we have any
        if ruler_patterns:
            self.entity_ruler.add_patterns(ruler_patterns)
    
    def identify_phi(self, text: str) -> List[Dict]:
        """
        Identify potential PHI elements in the given text.
        
        Args:
            text: The text to analyze for PHI
            
        Returns:
            List of dictionaries containing PHI findings with details
        """
        findings = []
        
        # Use Presidio for PHI detection if available
        if self.presidio_analyzer:
            # Analyze with Presidio
            presidio_results = self.presidio_analyzer.analyze(
                text=text,
                language=self.language,
            )
            
            # Convert Presidio results to our standard format
            for result in presidio_results:
                findings.append({
                    'start': result.start,
                    'end': result.end,
                    'type': result.entity_type,
                    'text': text[result.start:result.end],
                    'confidence': result.score,
                    'source': 'presidio'
                })
        
        # Use Scrubadub for PHI detection if available
        if self.scrubber:
            # Get filth from scrubadub
            filths = list(self.scrubber.iter_filth(text))
            
            # Convert scrubadub results to our standard format
            for filth in filths:
                # Scrubadub filth objects use 'beg' and 'end' rather than 'start' and 'end'
                filth_start = getattr(filth, 'beg', None)
                filth_end = getattr(filth, 'end', None)
                
                # If beg/end attributes aren't available, get the locations from match objects
                if filth_start is None or filth_end is None:
                    # Get indices from the text attribute
                    filth_text = getattr(filth, 'text', '')
                    if filth_text:
                        filth_start = text.find(filth_text)
                        if filth_start >= 0:
                            filth_end = filth_start + len(filth_text)
                        else:
                            # Skip if we can't locate the text
                            continue
                
                # Avoid duplicates from multiple engines
                is_duplicate = False
                for existing in findings:
                    if (existing['start'] == filth_start and 
                        existing['end'] == filth_end):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    findings.append({
                        'start': filth_start,
                        'end': filth_end,
                        'type': filth.type,
                        'text': text[filth_start:filth_end] if (filth_start is not None and filth_end is not None) else getattr(filth, 'text', ''),
                        'confidence': getattr(filth, 'confidence', 0.8),
                        'source': 'scrubadub'
                    })
        
        # Add custom regex pattern detection
        if self.use_regex:
            # Get all regex patterns from the RegexPatterns class
            pattern_dict = {
                attr: getattr(RegexPatterns, attr) 
                for attr in dir(RegexPatterns) 
                if not attr.startswith('__') and isinstance(getattr(RegexPatterns, attr), str)
            }
            
            # Apply each pattern
            for pattern_name, pattern in pattern_dict.items():
                for match in re.finditer(pattern, text):
                    # Map pattern names to PHI types
                    if "SSN" in pattern_name:
                        phi_type = "SSN"
                    elif "PHONE" in pattern_name:
                        phi_type = "PHONE_NUMBER"
                    elif "EMAIL" in pattern_name:
                        phi_type = "EMAIL_ADDRESS"
                    elif "POSTAL" in pattern_name or "ZIP" in pattern_name:
                        phi_type = "POSTAL_CODE"
                    elif "MEDICAL_RECORD" in pattern_name or "MRN" in pattern_name:
                        phi_type = "MEDICAL_RECORD_NUMBER"
                    elif "HEALTH" in pattern_name:
                        phi_type = "HEALTH_ID"
                    elif "SIN" in pattern_name:
                        phi_type = "SIN"
                    elif "DATE" in pattern_name:
                        phi_type = "DATE"
                    elif "CREDIT" in pattern_name:
                        phi_type = "CREDIT_CARD"
                    elif "LICENSE" in pattern_name:
                        phi_type = "LICENSE_NUMBER"
                    elif "PASSPORT" in pattern_name:
                        phi_type = "PASSPORT_NUMBER"
                    elif "IP" in pattern_name:
                        phi_type = "IP_ADDRESS"
                    elif "URL" in pattern_name:
                        phi_type = "URL"
                    else:
                        phi_type = "OTHER_ID"
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing in findings:
                        if (existing['start'] == match.start() and 
                            existing['end'] == match.end()):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        findings.append({
                            'start': match.start(),
                            'end': match.end(),
                            'type': phi_type,
                            'text': match.group(),
                            'confidence': 0.9,  # High confidence for regex matches
                            'source': 'regex'
                        })
        
        # Add enhanced detection using spaCy and patterns
        self._add_enhanced_detection(text, findings)
        
        # Sort findings by start position
        findings.sort(key=lambda x: x['start'])
        
        return findings
    
    def _add_enhanced_detection(self, text, findings):
        """Add enhanced detection using spaCy and patterns from patterns.json"""
        if not text or not ENHANCED_DETECTION_AVAILABLE or not self.nlp:
            return
            
        # Check for abusive language
        if self.profanity and self.profanity.contains_profanity(text):
            # Get all words
            words = text.split()
            # Check each word
            for i, word in enumerate(words):
                if self.profanity.contains_profanity(word):
                    # Find the position of this word in the original text
                    start_pos = 0
                    for j in range(i):
                        start_pos = text.find(words[j], start_pos) + len(words[j])
                        # Skip any whitespace
                        while start_pos < len(text) and text[start_pos].isspace():
                            start_pos += 1
                    
                    # Find the actual word in the text
                    word_pos = text.find(word, start_pos)
                    if word_pos >= 0:
                        # Check for duplicates
                        is_duplicate = False
                        for existing in findings:
                            if (existing['start'] == word_pos and 
                                existing['end'] == word_pos + len(word)):
                                is_duplicate = True
                                break
                                
                        if not is_duplicate:
                            findings.append({
                                'start': word_pos,
                                'end': word_pos + len(word),
                                'type': 'INAPPROPRIATE_LANGUAGE',
                                'text': word,
                                'confidence': 0.9,
                                'source': 'enhanced'
                            })
        
        # Use enhanced patterns from patterns.json
        for pattern in self.patterns:
            regex = pattern.get('regex', '')
            if not regex:
                continue
                
            for match in re.finditer(regex, text):
                # Check for duplicates
                is_duplicate = False
                for existing in findings:
                    if (existing['start'] == match.start() and 
                        existing['end'] == match.end()):
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    findings.append({
                        'start': match.start(),
                        'end': match.end(),
                        'type': pattern.get('category', 'PHI'),
                        'text': match.group(),
                        'confidence': 0.9,
                        'source': 'enhanced'
                    })
        
        # Use spaCy NER for additional entity detection
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'GPE', 'ORG', 'DATE', 'TIME', 'MONEY', 'QUANTITY', 'LOC', 'PRODUCT']:
                    # Check for duplicates
                    is_duplicate = False
                    for existing in findings:
                        if (existing['start'] == ent.start_char and 
                            existing['end'] == ent.end_char):
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        findings.append({
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'type': ent.label_,
                            'text': ent.text,
                            'confidence': 0.8,
                            'source': 'enhanced'
                        })
        except Exception as e:
            logger.warning(f"Error using spaCy NER: {e}")
    
    def check_malicious_content(self, text):
        """
        Check for malicious content patterns
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with detection results
        """
        if not self.malicious_patterns or not text:
            return {'malicious_detected': False, 'detections': [], 'max_score': 0}
            
        detections = []
        max_score = 0
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            regex = pattern.get('regex', '')
            if not regex:
                continue
                
            matches = list(re.finditer(regex, text))
            for match in matches:
                score = pattern.get('score', 1)
                max_score = max(max_score, score)
                detections.append({
                    'type': pattern.get('name', 'Unknown'),
                    'category': pattern.get('category', 'Malicious'),
                    'text': match.group(),
                    'score': score
                })
        
        # Check for abusive language if profanity filter is available
        if ENHANCED_DETECTION_AVAILABLE and self.profanity and self.profanity.contains_profanity(text):
            words = text.split()
            for word in words:
                if self.profanity.contains_profanity(word):
                    score = 3  # Default score for profanity
                    max_score = max(max_score, score)
                    detections.append({
                        'type': 'Profanity',
                        'category': 'Abusive',
                        'text': word,
                        'score': score
                    })
        
        return {
            'malicious_detected': len(detections) > 0,
            'detections': detections,
            'max_score': max_score
        }
    
    def replace_phi(self, text: str, replacement_format: str = '[{type}]') -> str:
        """
        Replace PHI in the text with the specified replacement format.
        
        Args:
            text: Text containing PHI to be replaced
            replacement_format: Format string for replacements (e.g. '[{type}]')
            
        Returns:
            Text with PHI replaced according to the format
        """
        if not text:
            return text
        
        # First try to use Presidio for replacement if available
        presidio_result = None
        if self.presidio_analyzer and self.presidio_anonymizer:
            analyzer_results = self.presidio_analyzer.analyze(text=text, language=self.language)
            
            # Only proceed with Presidio if it found entities
            if analyzer_results:
                # Build custom operator configs for formatting
                operators = {}
                for entity in set(result.entity_type for result in analyzer_results):
                    operators[entity] = OperatorConfig(
                        operator_name="replace", 
                        params={"new_value": replacement_format.format(type=entity)}
                    )
                
                anonymized_result = self.presidio_anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators=operators
                )
                
                presidio_result = anonymized_result.text
        
        # Get all findings from our comprehensive detection
        findings = self.identify_phi(text)
        
        # If Presidio found something and we have findings, merge the results
        if presidio_result and findings:
            # Find PHI missed by Presidio but found by our methods
            presidio_missed = []
            for finding in findings:
                # Check if this finding was missed in the Presidio result
                found_text = finding['text']
                if found_text in presidio_result and found_text in text:
                    # This was probably replaced by Presidio
                    continue
                
                presidio_missed.append(finding)
            
            # If Presidio missed some PHI, apply our replacements to the Presidio result
            if presidio_missed:
                result = presidio_result
                for finding in sorted(presidio_missed, key=lambda x: x['start'], reverse=True):
                    # Need to recalculate positions in the modified text
                    new_text = finding['text']
                    new_pos = result.find(new_text)
                    if new_pos >= 0:
                        replacement = replacement_format.format(type=finding['type'])
                        result = result[:new_pos] + replacement + result[new_pos + len(new_text):]
                return result
            else:
                # Presidio caught everything
                return presidio_result
        
        # If we don't have Presidio results or it didn't find anything, use our findings
        if findings:
            # Replace PHI in reverse order to avoid offset issues
            result = text
            for finding in sorted(findings, key=lambda x: x['start'], reverse=True):
                replacement = replacement_format.format(type=finding['type'])
                result = result[:finding['start']] + replacement + result[finding['end']:]
            return result
        else:
            # No PHI found by any method
            return text
    
    def redact_phi(self, text: str) -> str:
        """
        Completely redact (remove) PHI from text.
        
        Args:
            text: Text containing PHI to be redacted
            
        Returns:
            Text with PHI completely removed
        """
        return self.replace_phi(text, replacement_format="")
    
    def get_phi_report(self, text: str) -> Dict:
        """
        Generate a detailed report about PHI found in the text.
        
        Args:
            text: Text to analyze for PHI
            
        Returns:
            Dictionary with PHI statistics and details
        """
        findings = self.identify_phi(text)
        
        # Count occurrences by type
        type_counts = {}
        for finding in findings:
            phi_type = finding['type']
            if phi_type not in type_counts:
                type_counts[phi_type] = 0
            type_counts[phi_type] += 1
        
        # Check for malicious content
        malicious_report = self.check_malicious_content(text)
        malicious_content = malicious_report.get('malicious_detected', False)
        malicious_score = malicious_report.get('max_score', 0)
        
        # Build report
        report = {
            'total_phi_count': len(findings),
            'phi_types': type_counts,
            'phi_instances': findings,
            'phi_density': len(findings) / len(text) if text else 0,
            'compliance_risk': 'High' if findings else 'Low',
            'malicious_content': malicious_content,
            'malicious_score': malicious_score,
            'malicious_detections': malicious_report.get('detections', [])
        }
        
        return report

# Additional regular expressions for specialized PHI detection
class RegexPatterns:
    """Collection of regex patterns for PHI detection."""
    
    # US patterns
    SSN = r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
    US_PHONE = r'\b(\+?1[-\s]?)?(\([0-9]{3}\)|[0-9]{3})[-\s]?[0-9]{3}[-\s]?[0-9]{4}\b'
    US_ZIP = r'\b\d{5}([-\s]\d{4})?\b'
    
    # Canadian patterns
    CANADIAN_SIN = r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b'
    CANADIAN_POSTAL = r'\b[A-Za-z]\d[A-Za-z][\s-]?\d[A-Za-z]\d\b'
    CANADIAN_PHONE = r'\b(\+?1[-\s]?)?(\([0-9]{3}\)|[0-9]{3})[-\s]?[0-9]{3}[-\s]?[0-9]{4}\b'
    
    # Common healthcare identifiers
    MEDICAL_RECORD = r'\b(MR|MRN)[-\s]?\d{5,10}\b'
    HEALTH_CARD = r'\b\d{10}[A-Za-z]{2}\b'  # Ontario format
    HEALTH_INSURANCE = r'\b\d{9,12}\b'  # Generic health insurance ID pattern
    
    # Common email pattern
    EMAIL = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Additional patterns for comprehensive coverage
    CREDIT_CARD = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    DATE_PATTERN = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
    IP_ADDRESS = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    URL = r'\b(https?://|www\.)[^\s]+\b'
    PASSPORT = r'\b[A-Z]{1,2}\d{6,9}\b'
    DRIVER_LICENSE = r'\b[A-Z]\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{3}\b'

# Example usage
if __name__ == "__main__":
    # Example text with PHI
    sample_text = """
    Patient John Smith (DOB: 01/02/1945) was seen on 03/04/2023.
    His SSN is 123-45-6789 and his phone number is (555) 123-4567.
    He lives at 123 Main St, Anytown, NY 12345.
    His email is john.smith@example.com and MRN is MR-12345.
    His Ontario Health Card number is 1234567890AB.
    His credit card number is 4111-1111-1111-1111 and 
    IP address is 192.168.1.1.
    <script>alert('XSS attack');</script>
    This sentence contains a bad word: fuck.
    """
    
    # Create privacy filter
    privacy_filter = PrivacyFilter(use_presidio=True, use_scrubadub=True, use_regex=True)
    
    # Identify PHI
    phi_findings = privacy_filter.identify_phi(sample_text)
    print(f"Found {len(phi_findings)} PHI elements")
    
    # Replace PHI with tokens
    anonymized_text = privacy_filter.replace_phi(sample_text)
    print("\nAnonymized text:")
    print(anonymized_text)
    
    # Generate PHI report
    phi_report = privacy_filter.get_phi_report(sample_text)
    print("\nPHI Report:")
    print(f"Total PHI instances: {phi_report['total_phi_count']}")
    print(f"PHI types: {phi_report['phi_types']}")
    print(f"Compliance risk: {phi_report['compliance_risk']}")
    print(f"Malicious content detected: {phi_report['malicious_content']}")
    if phi_report['malicious_content']:
        print(f"Malicious score: {phi_report['malicious_score']}")
        print("Malicious detections:")
        for detection in phi_report['malicious_detections']:
            print(f"  - {detection['type']}: '{detection['text']}', Score: {detection['score']}")