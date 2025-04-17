from typing import List


def generate_open_exploration_prompt(seed: str) -> str:

    prompt = f"""
You're provided with a healthcare setting/specialty. Review it and respond with an output as requested below.

Healthcare Setting/Specialty:
{seed}

Requested Output instructions:
1. Identify a fundamental ontological category that can form a seed.
2. Build out the ontology tree one logical branch at a time. Add labels that form the next natural classification.
3. Add specific research areas at the 3rd layer.

For context, our focus is:
Quantitative research that can be reframed as intervention/control or cohorts for comparative study protocol design. Research topics can be clinical and/or operational in relation to the setting. No need to add patient-related details such as ID, age, etc. We only want a higher-order ontology that can be used to direct quantitative research hypotheses.

Format the output as JSON. Fill the nodes and links. Importantly, add hierarchy as requested. The tree must be comprehensive, covering the most common and novel topics in the area. Make sure the output is centered on the setting/specialty.

Final output should be formatted as:
```json
{{
  "nodes": [
    {{ "id": "...", "name": "...", "group": "..." }},
    ...
  ],
  "links": [
    {{ "source": "...", "target": "..." }},
    ...
  ]
}}
```

    Only JSON output is permitted. Comments in JSON or other explanations are not permitted.
    """
    return prompt

def generate_hypothesis_extend_prompt(graph_data: str, node_name: str) -> str:


    prompt = f"""
You are provided with graph data and a node name. Propose 1 new branch and 2-4 new nodes that contribute directly to the ontological category of the provided node.

Graph data:
{graph_data}

Node name:
{node_name}

Ensure that the output is in JSON format with updated nodes and links.

Final output should be formatted as:

{{
  "nodes": [
    {{ "id": "...", "name": "...", "group": "..." }},
    ...
  ],
  "links": [
    {{ "source": "...", "target": "..." }},
    ...
  ]
}}
Only JSON output is permitted. Comments in JSON or other explanations are not permitted.
"""
    return prompt

def generate_hypothesis_build_prompt(isolated_ontology: str, full_ontology: str) -> str:


    prompt = f"""
You're provided with ontologies that describe knowledge graph focused on a healthcare setting. 

There are 2 types of ontologies: 
Isolated ontology:
Focused on a specific branch of the knowledge graph

Full ontology:
Focused on the entire knowledge graph. 

You're requested to propose research hypotheses for comparative study protocol designs. Propose a json object with 1 hypothesis directly focused on isolated ontology and 4 hypotheses on full ontology. 

2-3 sentence research hypothesis for comparative study protocol design in the area of the provided ontology.

Isolated ontology:
{isolated_ontology}

Full ontology:
{full_ontology}

Format your response as sample below:
```json
{{
  "hypothesis_based_on_isolated_ontology": {{
      "hypothesis": "Implementation of personalized treatment plans based on genetic markers improves outcomes in patients with rare genetic disorders compared to standard treatment protocols.",
      "primary_endpoint": "Disease-specific outcome measures",
      "intervention": "Personalized treatment plans",
      "secondary_outcomes": ["Quality of Life", "Treatment Adherence"],
      "tags": ["GeneticDisorders", "PersonalizedMedicine", "RareDiseases"],
      "social_determinants": ["Access to Specialized Care", "Health Literacy"]
    }},
  "hypotheses_based_on_full_ontology": [
    {{
      "category": "Integrated Care and Health Systems",
      "hypothesis": "Implementation of integrated care pathways for patients with multiple chronic conditions reduces hospital readmissions compared to traditional siloed care approaches.",
      "primary_endpoint": "30-day hospital readmission rates",
      "intervention": "Integrated care pathways",
      "secondary_outcomes": ["Patient Satisfaction", "Healthcare Costs"],
      "tags": ["IntegratedCare", "ChronicConditions", "Readmissions"],
      "social_determinants": ["Healthcare Access", "Social Support"]
    }},
    ...
  ]
}}
```

Only JSON output is permitted. Comments in JSON or other explanations are not permitted.
"""    
    return prompt

def generate_hypothesis_sampler_prompt(num_hypotheses: int, param_lines: List[str]) -> str:
    prompt = f"""
Generate {num_hypotheses} clear and testable research hypotheses for healthcare research based on the following parameters:

{chr(10).join(param_lines)}

Each hypothesis should be:
1. Specific and testable
2. Include clear comparisons where appropriate
3. Specify measurable outcomes
4. Define the target population
5. State the expected direction of effect
6. Be relevant to the specified research context

Format the output as a JSON array where each object has the following structure:
{{
    "hypothesis": "A clear, testable hypothesis statement that includes the specific intervention/comparison, target population, and expected outcome",
    "study_type": "The most appropriate study design (e.g., RCT, cohort, case-control)"
}}

    Only provide JSON output, no additional text or explanations.
    """
    return prompt