from typing import Dict, List, Optional, Set
from protocols.model_calls import generate_protocol_sections
from protocols.template_helpers import ProtocolTemplateManager

class ProtocolBuilder:
    """Handles the creation and management of protocol sections"""
    
    def __init__(self):
        self.template_manager = ProtocolTemplateManager()
    
    @staticmethod
    def get_standard_sections() -> Dict[str, str]:
        """Returns standard protocol sections with descriptions"""
        template_manager = ProtocolTemplateManager()
        sections = {}
        for section_id, template in template_manager.templates['section_templates'].items():
            sections[template['title']] = template['prompts'][0] if template['prompts'] else ""
        return sections

    @staticmethod
    def get_initial_sections() -> List[Dict[str, str]]:
        """Returns initial sections to populate a new protocol"""
        template_manager = ProtocolTemplateManager()
        initial_sections = ['background', 'study_design']
        sections = []
        
        for section_id in initial_sections:
            try:
                template = template_manager.get_section_template(section_id)
                sections.append({
                    "title": template['title'],
                    "content": template_manager.generate_section_outline(section_id)
                })
            except ValueError:
                print(f"Warning: Could not find template for section {section_id}")
                
        return sections

    @staticmethod
    def create_section(title: str, content: str = "") -> Dict[str, any]:
        """Creates a new protocol section with metadata"""
        return {
            "title": title,
            "content": content,
            "version": 1,
            "contributors": {
                "human": {},
                "ai": {}
            }
        }

    @staticmethod
    async def generate_ai_protocol_sections(
        study_description: str,
        selected_sections: Set[str],
        literature_review: str = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate protocol sections using AI given a study description and optional literature review
        
        Args:
            study_description: Description of the study
            selected_sections: Set of section titles to generate
            literature_review: Optional literature review text
            
        Returns:
            Dictionary mapping section titles to subsection content
        """
        try:
            # Call AI model to generate protocol sections
            sections = await generate_protocol_sections(
                study_description=study_description,
                selected_sections=selected_sections,
                literature_review=literature_review
            )
            
            # The model should now generate exactly the sections we want
            # Just verify we got what we asked for
            if missing := (selected_sections - set(sections.keys())):
                print(f"Warning: Failed to generate some requested sections: {missing}")
                
            return sections
            
        except Exception as e:
            raise Exception(f"Failed to generate protocol sections: {str(e)}")
