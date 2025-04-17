from typing import Dict, List, Tuple
import difflib
from datetime import datetime

class ProtocolRecordKeeper:
    """Handles version control and contribution tracking for protocol sections"""
    
    @staticmethod
    def track_changes(old_text: str, new_text: str, char_contributors: List[Tuple], 
                     current_contributor: Tuple[str, str], paste_range: Tuple[int, int] = None) -> List[Tuple]:
        """
        Track character-level changes using difflib
        Returns updated char_contributors list
        """
        if old_text == new_text:
            return char_contributors
            
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        new_char_contributors = []
        current_pos = 0
        
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                new_char_contributors.extend(char_contributors[i1:i2])
                current_pos += (i2 - i1)
            elif op in ('insert', 'replace'):
                contrib_type, contrib_id = current_contributor
                for _ in range(j2 - j1):
                    new_char_contributors.append((current_pos, contrib_type, contrib_id))
                    current_pos += 1
                    
        # Handle paste events
        if paste_range:
            start, end = paste_range
            contrib_type, contrib_id = current_contributor
            for i in range(start, min(end, len(new_char_contributors))):
                new_char_contributors[i] = (i, contrib_type, contrib_id)
                
        return new_char_contributors

    @staticmethod
    def update_contributor_stats(char_contributors: List[Tuple]) -> Dict[str, Dict[str, int]]:
        """Recalculate contributor statistics based on character tagging"""
        contributors = {'human': {}, 'ai': {}}
        for _, contrib_type, contrib_id in char_contributors:
            if contrib_type == 'human':
                contributors['human'][contrib_id] = contributors['human'].get(contrib_id, 0) + 1
            else:
                contributors['ai'][contrib_id] = contributors['ai'].get(contrib_id, 0) + 1
        return contributors

    @staticmethod
    def create_version_entry(version_num: int, content: str) -> Dict:
        """Create a new version entry with metadata"""
        return {
            "version": version_num,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
