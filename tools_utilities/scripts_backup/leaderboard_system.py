
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List

class CompetitionLeaderboard:
    """Manages the competition leaderboard"""
    
    def __init__(self, leaderboard_path: str = "leaderboard/leaderboard.csv"):
        self.leaderboard_path = leaderboard_path
        self.leaderboard = self._load_leaderboard()
    
    def _load_leaderboard(self) -> pd.DataFrame:
        """Load existing leaderboard or create new one"""
        try:
            return pd.read_csv(self.leaderboard_path)
        except FileNotFoundError:
            return pd.DataFrame(columns=[
                'team_name', 'submission_id', 'timestamp', 'overall_score',
                'biological_accuracy', 'consciousness_emergence', 
                'computational_efficiency', 'generalization_score', 'innovation_score'
            ])
    
    def add_submission(self, team_name: str, submission_id: str, metrics: Dict[str, float]):
        """Add a new submission to the leaderboard"""
        submission = {
            'team_name': team_name,
            'submission_id': submission_id,
            'timestamp': datetime.now().isoformat(),
            'overall_score': metrics.get('overall_score', 0),
            'biological_accuracy': metrics.get('biological_accuracy', 0),
            'consciousness_emergence': metrics.get('consciousness_emergence', 0),
            'computational_efficiency': metrics.get('computational_efficiency', 0),
            'generalization_score': metrics.get('generalization_score', 0),
            'innovation_score': metrics.get('innovation_score', 0)
        }
        
        self.leaderboard = pd.concat([self.leaderboard, pd.DataFrame([submission])], ignore_index=True)
        self.leaderboard = self.leaderboard.sort_values('overall_score', ascending=False)
        self._save_leaderboard()
        
        print(f"✅ Submission from {team_name} added to leaderboard")
        print(f"Overall Score: {metrics.get('overall_score', 0):.4f}")
    
    def get_top_submissions(self, n: int = 10) -> pd.DataFrame:
        """Get top n submissions"""
        return self.leaderboard.head(n)
    
    def get_team_best(self, team_name: str) -> pd.DataFrame:
        """Get best submission for a specific team"""
        team_submissions = self.leaderboard[self.leaderboard['team_name'] == team_name]
        return team_submissions.head(1)
    
    def _save_leaderboard(self):
        """Save leaderboard to file"""
        self.leaderboard.to_csv(self.leaderboard_path, index=False)
    
    def display_leaderboard(self, n: int = 10):
        """Display formatted leaderboard"""
        top_submissions = self.get_top_submissions(n)
        print("🏆 Brain Simulation Challenge Leaderboard")
        print("=" * 80)
        print(f"{'Rank':<4} {'Team':<20} {'Overall':<8} {'Bio Acc':<8} {'Conscious':<8} {'Comp Eff':<8}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(top_submissions.iterrows(), 1):
            print(f"{i:<4} {row['team_name']:<20} {row['overall_score']:<8.4f} "
                  f"{row['biological_accuracy']:<8.4f} {row['consciousness_emergence']:<8.4f} "
                  f"{row['computational_efficiency']:<8.4f}")

# Usage example:
# leaderboard = CompetitionLeaderboard()
# leaderboard.add_submission("Team Quark", "submission_001", metrics)
# leaderboard.display_leaderboard()
