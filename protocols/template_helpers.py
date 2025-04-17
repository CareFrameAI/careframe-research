import json
from typing import Dict, List, Optional, Set
from pathlib import Path

class ProtocolTemplateManager:
    """Helper class for working with protocol templates"""
    
    def __init__(self, template_path: Optional[str] = None):
        """Initialize with optional custom template path"""
        if template_path is None:
            template_path = str(Path(__file__).parent / 'templates.json')
        self.template_path = template_path
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict:
        """Load templates from JSON file"""
        with open(self.template_path, 'r') as f:
            return json.load(f)
            
    def _normalize_section_id(self, section_id: str) -> str:
        """Normalize section ID to standard format"""
        return section_id.lower().replace(' ', '_').replace('-', '_')
            
    def get_study_types(self) -> List[Dict[str, str]]:
        """Get list of available study types with names and descriptions"""
        return [
            {
                'id': study_id,
                'name': info['name'],
                'description': info['description']
            }
            for study_id, info in self.templates['study_types'].items()
        ]
        
    def get_required_sections(self, study_type: str) -> List[str]:
        """Get required sections for a study type"""
        if study_type not in self.templates['study_types']:
            raise ValueError(f"Unknown study type: {study_type}")
        return self.templates['study_types'][study_type]['required_sections']
        
    def get_optional_sections(self, study_type: str) -> List[str]:
        """Get optional sections for a study type"""
        if study_type not in self.templates['study_types']:
            raise ValueError(f"Unknown study type: {study_type}")
        return self.templates['study_types'][study_type]['optional_sections']
        
    def get_section_template(self, section_id: str) -> Dict:
        """Get template for a specific section"""
        normalized_id = self._normalize_section_id(section_id)
        
        # Check standard sections
        if normalized_id in self.templates['section_templates']:
            return self.templates['section_templates'][normalized_id]
            
        # Check special sections
        if normalized_id in self.templates['special_sections']:
            return self.templates['special_sections'][normalized_id]
            
        # If not found, provide helpful error message
        all_sections = set(self.templates['section_templates'].keys()) | set(self.templates['special_sections'].keys())
        similar_sections = [s for s in all_sections if normalized_id in s or s in normalized_id]
        
        error_msg = f"Unknown section: {section_id}"
        if similar_sections:
            error_msg += f"\nDid you mean one of these? {', '.join(similar_sections)}"
            
        raise ValueError(error_msg)
        
    def get_section_prompts(self, section_id: str) -> List[str]:
        """Get prompts for a specific section"""
        return self.get_section_template(section_id)['prompts']
        
    def get_section_subsections(self, section_id: str) -> List[str]:
        """Get subsections for a specific section"""
        return self.get_section_template(section_id)['subsections']
        
    def get_all_sections_for_study(self, study_type: str) -> Dict[str, bool]:
        """Get all available sections for a study type with required flag"""
        if study_type not in self.templates['study_types']:
            raise ValueError(f"Unknown study type: {study_type}")
            
        required = set(self.get_required_sections(study_type))
        optional = set(self.get_optional_sections(study_type))
        
        return {
            section: section in required
            for section in required | optional
        }
        
    def validate_section_content(self, section_id: str, content: Dict) -> List[str]:
        """
        Validate section content against template
        Returns list of missing required subsections
        """
        template = self.get_section_template(section_id)
        required_subsections = set(template['subsections'])
        provided_subsections = set(content.keys())
        
        return list(required_subsections - provided_subsections)
        
    def generate_section_outline(self, section_id: str) -> Dict[str, str]:
        """Generate empty outline for a section with all subsections"""
        template = self.get_section_template(section_id)
        return {
            subsection: ""
            for subsection in template['subsections']
        }
        
    def get_section_guidance(self, section_id: str) -> Dict[str, List[str]]:
        """Get comprehensive guidance for writing a section"""
        template = self.get_section_template(section_id)
        return {
            'title': template['title'],
            'subsections': template['subsections'],
            'prompts': template['prompts']
        }
        
    def suggest_next_sections(self, study_type: str, completed_sections: Set[str]) -> List[str]:
        """Suggest next sections to complete based on study type and current progress"""
        required = set(self.get_required_sections(study_type))
        optional = set(self.get_optional_sections(study_type))
        all_sections = required | optional
        
        # First suggest uncompleted required sections
        suggestions = list(required - completed_sections)
        
        # Then suggest optional sections if all required are done
        if not suggestions:
            suggestions = list(optional - completed_sections)
            
        return suggestions
        
    def get_section_dependencies(self, section_id: str) -> List[str]:
        """Get recommended sections to complete before this one"""
        # Define common dependencies
        dependencies = {
            'statistical_analysis': ['objectives', 'outcomes'],
            'randomization': ['population', 'study_design'],
            'interventions': ['study_design'],
            'outcomes': ['objectives'],
            'sample_size': ['outcomes', 'statistical_analysis'],
            'adaptation_rules': ['interim_analyses', 'statistical_analysis'],
            'cost_effectiveness': ['outcomes', 'interventions']
        }
        return dependencies.get(section_id, [])
        
    def export_study_template(self, study_type: str, format: str = 'markdown') -> str:
        """Export study template in specified format"""
        if study_type not in self.templates['study_types']:
            raise ValueError(f"Unknown study type: {study_type}")
            
        study_info = self.templates['study_types'][study_type]
        sections = self.get_all_sections_for_study(study_type)
        
        if format == 'markdown':
            output = [
                f"# {study_info['name']} Protocol Template\n",
                f"## Description\n{study_info['description']}\n",
                "## Sections\n"
            ]
            
            for section_id, is_required in sections.items():
                template = self.get_section_template(section_id)
                status = "Required" if is_required else "Optional"
                output.extend([
                    f"### {template['title']} ({status})\n",
                    "#### Subsections\n" + 
                    "\n".join(f"- {s}" for s in template['subsections']) + "\n",
                    "#### Guidance\n" +
                    "\n".join(f"- {p}" for p in template['prompts']) + "\n"
                ])
                
            return "\n".join(output)
        else:
            raise ValueError(f"Unsupported export format: {format}") 