import json
from typing import Dict, Optional
from protocols.prompts import protocol_sections_prompt
from protocols.template_helpers import ProtocolTemplateManager
from llms.client import call_llm_async

async def generate_protocol_sections(
    study_description: str,
    selected_sections: set[str],
    literature_review: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Generate protocol sections using AI model
    
    Args:
        study_description: Description of the study
        selected_sections: Set of section titles to generate
        literature_review: Optional literature review text
        
    Returns:
        Dictionary mapping section titles to subsection content dictionaries
        
    Raises:
        ValueError: If response parsing fails or sections are invalid
        Exception: For other generation errors
    """
    template_manager = ProtocolTemplateManager()
    prompt = protocol_sections_prompt(study_description, selected_sections, literature_review)
    
    try:
        response = await call_llm_async(prompt)
        # Clean up response to ensure valid JSON
        json_str = response.replace("```json", "").replace("```", "").strip()
        # Remove any trailing commas that might break JSON parsing
        json_str = json_str.replace(",\n}", "\n}")
        json_str = json_str.replace(",\n]", "\n]")
        # Remove comments that might break JSON parsing
        json_str = "\n".join(line for line in json_str.split("\n") 
                            if not line.strip().startswith("//"))
        
        sections = json.loads(json_str)
        
        # Validate section structure and check for missing sections
        received_sections = set(sections.keys())
        missing_sections = selected_sections - received_sections
        extra_sections = received_sections - selected_sections
        
        if missing_sections:
            print(f"Warning: Model did not generate these requested sections: {missing_sections}")
        if extra_sections:
            print(f"Warning: Model generated extra unrequested sections: {extra_sections}")
            # Remove extra sections
            for section in extra_sections:
                del sections[section]
        
        # Validate section structure
        for section_title, content in sections.items():
            if not isinstance(content, dict):
                raise ValueError(f"Invalid section structure for {section_title}: "
                               f"expected dictionary of subsections")
                
            # Validate against template if it's a standard section
            try:
                template = template_manager.get_section_template(section_title.lower())
                missing = template_manager.validate_section_content(
                    section_title.lower(), 
                    {k.lower(): v for k, v in content.items()}
                )
                if missing:
                    print(f"Warning: Missing subsections in {section_title}: {missing}")
            except ValueError:
                # Custom section - just verify it has some subsections
                if not content:
                    print(f"Warning: Custom section {section_title} has no subsections")
                
        return sections
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to generate protocol sections: {str(e)}")