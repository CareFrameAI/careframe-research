import json
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from llms.client import call_llm_async
from hypotheses.prompts import generate_hypothesis_sampler_prompt

class HypothesisSampler:
    def __init__(self):
        # Load ontology from JSON file
        ontology_path = Path(__file__).parent / "sampler_ontology.json"
        with open(ontology_path, 'r') as f:
            self.ontology = json.load(f)

    def _sample_from_dict(self, data: Dict[str, List[str]], n: int = 1) -> List[str]:
        """Sample from nested dictionary structure"""
        all_items = []
        for category in data.values():
            if isinstance(category, list):
                all_items.extend(category)
            elif isinstance(category, dict):
                all_items.extend(self._sample_from_dict(category))
        return random.sample(all_items, min(n, len(all_items)))

    def _sample_from_list(self, items: List[str], n: int = 1) -> List[str]:
        """Sample from list"""
        return random.sample(items, min(n, len(items)))

    def generate_seed_combination(self) -> Dict[str, str]:
        """Generate a random combination of elements from the ontology"""
        return {
            "research_type": self._sample_from_list(self.ontology["research_types"])[0],
            "department": self._sample_from_dict(self.ontology["departments"])[0],
            "intervention": self._sample_from_list(self.ontology["intervention_types"])[0],
            "outcome": self._sample_from_list(self.ontology["outcome_domains"])[0],
            "population": self._sample_from_list(self.ontology["population_focus"])[0],
            "intent": self._sample_from_list(self.ontology["research_intent"])[0]
        }

    async def generate_hypotheses_with_params(self, params: Dict[str, str], num_hypotheses: int = 5) -> List[Dict[str, Any]]:
        """Generate research hypotheses using specific parameters"""
        # Build prompt with only the parameters that were specified
        param_lines = []
        for param_name, display_name in [
            ('research_type', 'Research Type'),
            ('department', 'Department/Setting'),
            ('intervention', 'Intervention Type'),
            ('outcome', 'Outcome Domain'),
            ('population', 'Population Focus'),
            ('intent', 'Research Intent')
        ]:
            if param_name in params:
                param_lines.append(f"{display_name}: {params[param_name]}")

        prompt = generate_hypothesis_sampler_prompt(num_hypotheses, param_lines)

        try:
            response = await call_llm_async(prompt)
            hypotheses = json.loads(response)
            return hypotheses
        except Exception as e:
            raise Exception(f"Failed to generate hypotheses: {str(e)}")

    async def generate_hypotheses(self, num_hypotheses: int = 10) -> List[Dict[str, Any]]:
        """Generate research hypotheses using random parameters"""
        params = self.generate_seed_combination()
        return await self.generate_hypotheses_with_params(params, num_hypotheses)

