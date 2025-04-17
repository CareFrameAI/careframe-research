# exchange/blockchain/team_validator.py

import requests
from datetime import datetime

class TeamBasedValidator:
    """Validator implementation using teams and organizations structure"""
    
    def __init__(self, teams_db_url="http://localhost:5984", auth=("admin", "cfpwd")):
        self.teams_db_url = teams_db_url
        self.auth = auth
        self.validators = {}  # team_id -> validator_info
    
    def load_validators_from_teams(self):
        """Load validator data from teams database"""
        try:
            response = requests.get(
                f"{self.teams_db_url}/teams/_all_docs?include_docs=true",
                auth=self.auth
            )
            
            if response.status_code == 200:
                teams = [row['doc'] for row in response.json()['rows'] if 'doc' in row]
                
                for team in teams:
                    team_id = team.get('_id')
                    if team_id:
                        self.validators[team_id] = {
                            'team_name': team.get('name', 'Unknown Team'),
                            'organization_id': team.get('organization_id', 'independent'),
                            'members': team.get('members', []),
                            'weight': 1.0  # Default weight, can be customized
                        }
                
                return True
            return False
        except Exception as e:
            print(f"Error loading validators from teams: {e}")
            return False
    
    def get_validator_for_user(self, email):
        """Get validator ID for a given user email"""
        for team_id, validator in self.validators.items():
            if any(member.get('email') == email for member in validator['members']):
                return team_id
        return None
    
    def is_valid_validator(self, validator_id):
        """Check if validator ID is valid"""
        return validator_id in self.validators
    
    def get_validator_members(self, validator_id):
        """Get team members for a validator"""
        if validator_id in self.validators:
            return self.validators[validator_id].get('members', [])
        return []
    
    def get_admin_validators(self):
        """Get IDs of all teams with admin members"""
        admin_teams = []
        for team_id, validator in self.validators.items():
            if any(member.get('is_admin', False) for member in validator['members']):
                admin_teams.append(team_id)
        return admin_teams