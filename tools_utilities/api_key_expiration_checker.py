#!/usr/bin/env python3
"""
API Key Expiration Checker for Pre-Push Hook
Checks all API keys and fails if any are expired
"""

import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from api_key_manager import APIKeyManager, KeyStatus
except ImportError:
    print("❌ Error: api_key_manager module not found")
    sys.exit(1)

class PrePushAPIChecker:
    """API key expiration checker for pre-push hook"""
    
    def __init__(self):
        self.credentials_path = Path("/Users/camdouglas/quark/data/credentials/all_api_keys.json")
        self.ignore_file = Path("/Users/camdouglas/quark/.git/api_key_ignore")
        
        # Direct renewal links for each service
        self.renewal_links = {
            'materials_project': {
                'url': 'https://next-gen.materialsproject.org/api',
                'description': 'Materials Project API Dashboard'
            },
            'openai': {
                'url': 'https://platform.openai.com/api-keys',
                'description': 'OpenAI API Keys Management'
            },
            'claude': {
                'url': 'https://console.anthropic.com/',
                'description': 'Anthropic Console - API Keys'
            },
            'gemini': {
                'url': 'https://makersuite.google.com/app/apikey',
                'description': 'Google AI Studio - API Keys'
            },
            'github': {
                'url': 'https://github.com/settings/tokens',
                'description': 'GitHub Personal Access Tokens'
            },
            'huggingface': {
                'url': 'https://huggingface.co/settings/tokens',
                'description': 'Hugging Face Access Tokens'
            },
            'context7': {
                'url': 'https://context7.ai/dashboard/api-keys',
                'description': 'Context7 API Keys Dashboard'
            },
            'wolfram': {
                'url': 'https://developer.wolframalpha.com/portal/myapps/',
                'description': 'Wolfram Alpha Developer Portal'
            },
            'kaggle': {
                'url': 'https://www.kaggle.com/settings/account',
                'description': 'Kaggle Account Settings - API'
            }
        }
    
    def load_ignored_keys(self) -> List[str]:
        """Load list of ignored API keys"""
        if not self.ignore_file.exists():
            return []
        
        try:
            with open(self.ignore_file, 'r') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception:
            return []
    
    def save_ignored_keys(self, ignored_keys: List[str]):
        """Save list of ignored API keys"""
        try:
            with open(self.ignore_file, 'w') as f:
                f.write("# Temporarily ignored API keys for pre-push hook\n")
                f.write("# Format: service_name\n")
                f.write("# Remove lines to re-enable checking\n\n")
                for key in ignored_keys:
                    f.write(f"{key}\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not save ignore file: {e}")
    
    async def check_api_keys(self) -> Tuple[bool, List[Dict]]:
        """Check all API keys for expiration"""
        try:
            key_manager = APIKeyManager(self.credentials_path)
            ignored_keys = self.load_ignored_keys()
            
            # Check all keys
            statuses = await key_manager.check_all_keys()
            
            # Find problematic keys
            problems = []
            
            for service, status in statuses.items():
                if service in ignored_keys:
                    continue
                
                key_info = key_manager.key_info.get(service)
                if not key_info:
                    continue
                
                # Check for expiration
                if status == KeyStatus.EXPIRED:
                    problems.append({
                        'service': service,
                        'issue': 'expired',
                        'message': f"API key for {service} has EXPIRED",
                        'expires_at': key_info.expires_at.isoformat() if key_info.expires_at else None
                    })
                
                elif status == KeyStatus.INVALID:
                    problems.append({
                        'service': service,
                        'issue': 'invalid',
                        'message': f"API key for {service} is INVALID",
                        'expires_at': None
                    })
                
                elif key_info.needs_renewal(buffer_days=7):  # 7 day warning
                    days_left = (key_info.expires_at - datetime.now()).days
                    problems.append({
                        'service': service,
                        'issue': 'expiring_soon',
                        'message': f"API key for {service} expires in {days_left} days",
                        'expires_at': key_info.expires_at.isoformat()
                    })
            
            return len(problems) == 0, problems
            
        except Exception as e:
            print(f"❌ Error checking API keys: {e}")
            return False, [{'service': 'system', 'issue': 'error', 'message': str(e)}]
    
    def format_renewal_link(self, service: str) -> str:
        """Format renewal link for service"""
        if service in self.renewal_links:
            link_info = self.renewal_links[service]
            return f"🔗 {link_info['description']}: {link_info['url']}"
        else:
            return f"🔗 Search for '{service} API key renewal' in your browser"
    
    def print_problems(self, problems: List[Dict]):
        """Print formatted problem report"""
        print("\n" + "="*80)
        print("🚨 API KEY ISSUES DETECTED - PUSH BLOCKED")
        print("="*80)
        
        expired_keys = [p for p in problems if p['issue'] == 'expired']
        invalid_keys = [p for p in problems if p['issue'] == 'invalid']
        expiring_keys = [p for p in problems if p['issue'] == 'expiring_soon']
        
        if expired_keys:
            print(f"\n❌ EXPIRED KEYS ({len(expired_keys)}):")
            for problem in expired_keys:
                print(f"   • {problem['service']}: {problem['message']}")
                print(f"     {self.format_renewal_link(problem['service'])}")
        
        if invalid_keys:
            print(f"\n❌ INVALID KEYS ({len(invalid_keys)}):")
            for problem in invalid_keys:
                print(f"   • {problem['service']}: {problem['message']}")
                print(f"     {self.format_renewal_link(problem['service'])}")
        
        if expiring_keys:
            print(f"\n⚠️  EXPIRING SOON ({len(expiring_keys)}):")
            for problem in expiring_keys:
                print(f"   • {problem['service']}: {problem['message']}")
                print(f"     {self.format_renewal_link(problem['service'])}")
        
        print(f"\n💡 RESOLUTION OPTIONS:")
        print(f"   1. Renew expired/invalid keys using the links above")
        print(f"   2. Temporarily ignore keys: echo 'service_name' >> .git/api_key_ignore")
        print(f"   3. Force push (not recommended): git push --no-verify")
        
        print(f"\n📄 Ignore file location: {self.ignore_file}")
        print(f"📄 Credentials file: {self.credentials_path}")
        print("="*80)
    
    async def run_check(self) -> int:
        """Run the API key check and return exit code"""
        print("🔍 Checking API key expiration status...")
        
        success, problems = await self.check_api_keys()
        
        if success:
            print("✅ All API keys are valid and not expired")
            return 0
        else:
            self.print_problems(problems)
            return 1

async def main():
    """Main entry point for pre-push hook"""
    checker = PrePushAPIChecker()
    exit_code = await checker.run_check()
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())
