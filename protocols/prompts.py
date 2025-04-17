from protocols.template_helpers import ProtocolTemplateManager

def protocol_sections_prompt(study_description: str, selected_sections: set[str], literature_review: str = None):
    """Generates a study protocol given a clinical study description and optional literature review.
    
    Args:
        study_description: Description of the study
        selected_sections: Set of section titles to generate
        literature_review: Optional literature review text
    
    Returns:
        A prompt that will generate a JSON object where keys are protocol section titles and values 
        are dictionaries containing subsection content based on the template structure.
    """
    # Get template manager to access standard subsections
    template_manager = ProtocolTemplateManager()
    
    # Build section requirements list with subsections
    section_requirements = []
    for section in sorted(selected_sections):
        try:
            template = template_manager.get_section_template(section.lower())
            subsections = template['subsections']
            section_requirements.append(f"- {section}:\n  Required subsections: {', '.join(subsections)}")
        except ValueError:
            # Custom section
            section_requirements.append(f"- {section} (custom section)")
    
    sections_list = "\n".join(section_requirements)
    
    prompt = f"""
    Generate a clinical study protocol in JSON format based on the provided study description and literature review.
    The output should be a JSON object containing EXACTLY the following sections, where each section value is a dictionary 
    containing the specified subsections:

    Required Sections and their subsections:
    {sections_list}

    Study Description:
    {study_description}
    """
    
    if literature_review:
        prompt += f"""
    Literature Review:
    {literature_review}
        """

    prompt += """
    Required Output Structure Example:
    ```json
    {
        "Background": {
            "literature_review": "Comprehensive review of relevant literature...",
            "rationale": "Clear justification for conducting the study...",
            "preliminary_data": "Any preliminary work that supports this study...",
            "significance": "Impact and importance of the research..."
        },
        "Interventions": {
            "intervention_details": "Detailed description of interventions...",
            "control_group": "Description of control/comparison group...",
            "compliance": "Methods to monitor and ensure compliance...",
            "concomitant_care": "Permitted and prohibited concomitant care..."
        }
    }
    ```

    Notes:
    - The output must be valid JSON
    - Generate content for EXACTLY the sections listed above
    - Include ALL required subsections for each standard section
    - For custom sections, create 3-4 logical subsections based on the section title
    - Content should be detailed and specific to the study description
    - Incorporate literature review information where relevant
    - Use evidence-based language and maintain scientific tone
    - Follow standard protocol writing guidelines
    - Ensure consistency across sections
    """
    return prompt