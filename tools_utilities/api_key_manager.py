#!/usr/bin/env python3
"""
API Key Management System
Handles expiration, rotation, and auto-renewal for all validation sources
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
from enum import Enum

logger = logging.getLogger(__name__)

class KeyStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALID = "invalid"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"

@dataclass
class APIKeyInfo:
    """Information about an API key"""
    service: str
    key: str
    status: KeyStatus = KeyStatus.UNKNOWN
    expires_at: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_reset: Optional[datetime] = None
    auto_renewable: bool = False
    renewal_endpoint: Optional[str] = None
    renewal_method: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if key is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def needs_renewal(self, buffer_days: int = 7) -> bool:
        """Check if key needs renewal (within buffer period)"""
        if self.expires_at is None:
            return False
        return datetime.now() + timedelta(days=buffer_days) > self.expires_at

class APIKeyManager:
    """
    Manages API keys for all validation sources
    Handles expiration checking, rotation, and auto-renewal
    """
    
    def __init__(self, credentials_path: Path):
        self.credentials_path = credentials_path
        self.credentials = self._load_credentials()
        self.key_info: Dict[str, APIKeyInfo] = {}
        self.backup_keys: Dict[str, List[str]] = {}
        
        # Service-specific configurations
        self.service_configs = self._initialize_service_configs()
        
        # Initialize key info
        self._initialize_key_info()
        
        logger.info(f"✅ API Key Manager initialized with {len(self.key_info)} keys")
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials from file"""
        with open(self.credentials_path, 'r') as f:
            return json.load(f)
    
    def _save_credentials(self):
        """Save credentials back to file"""
        with open(self.credentials_path, 'w') as f:
            json.dump(self.credentials, f, indent=2)
    
    def _initialize_service_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize service-specific configurations"""
        return {
            # Scientific APIs (mostly no expiration)
            'arxiv': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'http://export.arxiv.org/api/query?search_query=test&max_results=1'
            },
            'pubmed': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi'
            },
            'pubchem': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/water/property/MolecularFormula/JSON'
            },
            'alphafold': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://alphafold.ebi.ac.uk/api/prediction/P21359'
            },
            'rcsb_pdb': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://data.rcsb.org/rest/v1/core/entry/4HHB'
            },
            'ensembl': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://rest.ensembl.org/info/ping'
            },
            'uniprot': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://rest.uniprot.org/uniprotkb/P04637'
            },
            
            # Commercial APIs (may expire)
            'materials_project': {
                'expires': True,
                'expiry_days': 365,  # Typically 1 year
                'auto_renewable': False,  # Manual renewal required
                'check_endpoint': 'https://api.materialsproject.org/materials/summary',
                'auth_header': 'X-API-KEY'
            },
            'openai': {
                'expires': False,  # No automatic expiration
                'auto_renewable': False,
                'check_endpoint': 'https://api.openai.com/v1/models',
                'auth_header': 'Authorization',
                'auth_prefix': 'Bearer '
            },
            'claude': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://api.anthropic.com/v1/messages',
                'auth_header': 'x-api-key'
            },
            'gemini': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://generativelanguage.googleapis.com/v1beta/models',
                'auth_param': 'key'
            },
            'wolfram': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'http://api.wolframalpha.com/v2/query',
                'auth_param': 'appid'
            },
            'huggingface': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://huggingface.co/api/whoami',
                'auth_header': 'Authorization',
                'auth_prefix': 'Bearer '
            },
            'github': {
                'expires': True,
                'expiry_days': 365,  # Personal access tokens can expire
                'auto_renewable': False,
                'check_endpoint': 'https://api.github.com/user',
                'auth_header': 'Authorization',
                'auth_prefix': 'token '
            },
            'kaggle': {
                'expires': False,
                'auto_renewable': False,
                'check_endpoint': 'https://www.kaggle.com/api/v1/datasets/list',
                'auth_header': 'Authorization',
                'auth_prefix': 'Basic '
            },
            'context7': {
                'expires': True,
                'expiry_days': 90,  # Typical for service APIs
                'auto_renewable': False,
                'check_endpoint': 'https://api.context7.ai/v1/health'
            }
        }
    
    def _initialize_key_info(self):
        """Initialize key information from credentials"""
        services = self.credentials.get('services', {})
        
        for service_name, service_data in services.items():
            # Get primary key
            key = service_data.get('api_key') or service_data.get('token') or service_data.get('app_id')
            
            if key:
                config = self.service_configs.get(service_name, {})
                
                # Calculate expiry if applicable
                expires_at = None
                if config.get('expires', False) and 'expiry_days' in config:
                    # Assume key was created recently if no creation date available
                    expires_at = datetime.now() + timedelta(days=config['expiry_days'])
                
                self.key_info[service_name] = APIKeyInfo(
                    service=service_name,
                    key=key,
                    expires_at=expires_at,
                    auto_renewable=config.get('auto_renewable', False),
                    renewal_endpoint=config.get('renewal_endpoint'),
                    renewal_method=config.get('renewal_method')
                )
                
                # Store backup keys if available
                backup_keys = service_data.get('backup_keys', [])
                if backup_keys:
                    self.backup_keys[service_name] = backup_keys
    
    async def check_key_status(self, service: str) -> KeyStatus:
        """Check the status of an API key"""
        if service not in self.key_info:
            return KeyStatus.UNKNOWN
        
        key_info = self.key_info[service]
        config = self.service_configs.get(service, {})
        
        # Check if expired by date
        if key_info.is_expired():
            key_info.status = KeyStatus.EXPIRED
            return KeyStatus.EXPIRED
        
        # Test key with actual API call
        try:
            status = await self._test_key_with_api(service, key_info.key)
            key_info.status = status
            key_info.last_checked = datetime.now()
            return status
            
        except Exception as e:
            logger.error(f"❌ Error checking key for {service}: {e}")
            key_info.status = KeyStatus.UNKNOWN
            return KeyStatus.UNKNOWN
    
    async def _test_key_with_api(self, service: str, key: str) -> KeyStatus:
        """Test API key with actual API call"""
        config = self.service_configs.get(service, {})
        check_endpoint = config.get('check_endpoint')
        
        if not check_endpoint:
            return KeyStatus.UNKNOWN
        
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare headers and params
                headers = {}
                params = {}
                
                # Add authentication
                if 'auth_header' in config:
                    auth_value = key
                    if 'auth_prefix' in config:
                        auth_value = config['auth_prefix'] + key
                    headers[config['auth_header']] = auth_value
                
                if 'auth_param' in config:
                    params[config['auth_param']] = key
                
                # Make test request
                async with session.get(
                    check_endpoint,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        return KeyStatus.ACTIVE
                    elif response.status == 401:
                        return KeyStatus.INVALID
                    elif response.status == 403:
                        return KeyStatus.INVALID
                    elif response.status == 429:
                        return KeyStatus.RATE_LIMITED
                    else:
                        return KeyStatus.UNKNOWN
                        
        except asyncio.TimeoutError:
            return KeyStatus.UNKNOWN
        except Exception as e:
            logger.error(f"❌ API test error for {service}: {e}")
            return KeyStatus.UNKNOWN
    
    async def check_all_keys(self) -> Dict[str, KeyStatus]:
        """Check status of all API keys"""
        logger.info("🔍 Checking status of all API keys...")
        
        results = {}
        
        # Check keys in parallel
        tasks = []
        for service in self.key_info.keys():
            task = asyncio.create_task(self.check_key_status(service))
            tasks.append((service, task))
        
        # Wait for all checks to complete
        for service, task in tasks:
            try:
                status = await task
                results[service] = status
                
                status_icon = {
                    KeyStatus.ACTIVE: "✅",
                    KeyStatus.EXPIRED: "⏰",
                    KeyStatus.INVALID: "❌",
                    KeyStatus.RATE_LIMITED: "⚠️",
                    KeyStatus.UNKNOWN: "❓"
                }
                
                logger.info(f"   {status_icon[status]} {service}: {status.value}")
                
            except Exception as e:
                logger.error(f"❌ Error checking {service}: {e}")
                results[service] = KeyStatus.UNKNOWN
        
        return results
    
    def rotate_to_backup_key(self, service: str) -> bool:
        """Rotate to backup key if available"""
        if service not in self.backup_keys or not self.backup_keys[service]:
            logger.warning(f"⚠️ No backup keys available for {service}")
            return False
        
        # Get current key
        current_key = self.key_info[service].key
        backup_keys = self.backup_keys[service]
        
        # Find a different backup key
        for backup_key in backup_keys:
            if backup_key != current_key:
                # Update credentials
                service_data = self.credentials['services'][service]
                
                # Store old key as backup
                if current_key not in backup_keys:
                    backup_keys.append(current_key)
                
                # Update to new key
                if 'api_key' in service_data:
                    service_data['api_key'] = backup_key
                elif 'token' in service_data:
                    service_data['token'] = backup_key
                elif 'app_id' in service_data:
                    service_data['app_id'] = backup_key
                
                # Update key info
                self.key_info[service].key = backup_key
                self.key_info[service].status = KeyStatus.UNKNOWN
                
                # Save credentials
                self._save_credentials()
                
                logger.info(f"✅ Rotated {service} to backup key")
                return True
        
        logger.warning(f"⚠️ No suitable backup key found for {service}")
        return False
    
    def get_expiring_keys(self, days_ahead: int = 30) -> List[Tuple[str, datetime]]:
        """Get keys that will expire within specified days"""
        expiring = []
        
        for service, key_info in self.key_info.items():
            if key_info.expires_at:
                days_until_expiry = (key_info.expires_at - datetime.now()).days
                if 0 <= days_until_expiry <= days_ahead:
                    expiring.append((service, key_info.expires_at))
        
        return sorted(expiring, key=lambda x: x[1])
    
    def generate_renewal_instructions(self) -> Dict[str, Dict[str, Any]]:
        """Generate instructions for manually renewing API keys"""
        instructions = {}
        
        renewal_info = {
            'materials_project': {
                'url': 'https://next-gen.materialsproject.org/api',
                'steps': [
                    '1. Log in to Materials Project website',
                    '2. Go to API section in your account',
                    '3. Generate new API key',
                    '4. Update credentials file'
                ],
                'documentation': 'https://next-gen.materialsproject.org/api'
            },
            'openai': {
                'url': 'https://platform.openai.com/api-keys',
                'steps': [
                    '1. Log in to OpenAI Platform',
                    '2. Go to API Keys section',
                    '3. Create new secret key',
                    '4. Update credentials file',
                    '5. Delete old key for security'
                ],
                'documentation': 'https://platform.openai.com/docs/quickstart'
            },
            'claude': {
                'url': 'https://console.anthropic.com/',
                'steps': [
                    '1. Log in to Anthropic Console',
                    '2. Go to API Keys section',
                    '3. Generate new API key',
                    '4. Update credentials file'
                ],
                'documentation': 'https://docs.anthropic.com/claude/reference/getting-started'
            },
            'github': {
                'url': 'https://github.com/settings/tokens',
                'steps': [
                    '1. Go to GitHub Settings > Developer settings > Personal access tokens',
                    '2. Generate new token (classic)',
                    '3. Select appropriate scopes',
                    '4. Update credentials file',
                    '5. Delete old token'
                ],
                'documentation': 'https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token'
            },
            'context7': {
                'url': 'https://context7.ai/dashboard',
                'steps': [
                    '1. Log in to Context7 dashboard',
                    '2. Go to API section',
                    '3. Generate new API key',
                    '4. Update credentials file'
                ],
                'documentation': 'https://context7.ai/docs'
            }
        }
        
        # Check which keys need renewal
        expiring_keys = self.get_expiring_keys(days_ahead=60)  # 60 day window
        
        for service, expiry_date in expiring_keys:
            if service in renewal_info:
                instructions[service] = {
                    **renewal_info[service],
                    'expires_at': expiry_date.isoformat(),
                    'days_until_expiry': (expiry_date - datetime.now()).days
                }
        
        return instructions
    
    def get_key_report(self) -> Dict[str, Any]:
        """Generate comprehensive API key report"""
        report = {
            'summary': {
                'total_keys': len(self.key_info),
                'active_keys': sum(1 for k in self.key_info.values() if k.status == KeyStatus.ACTIVE),
                'expired_keys': sum(1 for k in self.key_info.values() if k.status == KeyStatus.EXPIRED),
                'invalid_keys': sum(1 for k in self.key_info.values() if k.status == KeyStatus.INVALID),
                'unknown_keys': sum(1 for k in self.key_info.values() if k.status == KeyStatus.UNKNOWN)
            },
            'expiring_soon': self.get_expiring_keys(days_ahead=30),
            'renewal_instructions': self.generate_renewal_instructions(),
            'backup_keys_available': {
                service: len(keys) for service, keys in self.backup_keys.items()
            },
            'key_details': {
                service: {
                    'status': key_info.status.value,
                    'expires_at': key_info.expires_at.isoformat() if key_info.expires_at else None,
                    'last_checked': key_info.last_checked.isoformat() if key_info.last_checked else None,
                    'auto_renewable': key_info.auto_renewable,
                    'has_backup': service in self.backup_keys and len(self.backup_keys[service]) > 0
                }
                for service, key_info in self.key_info.items()
            }
        }
        
        return report

async def main():
    """Test API key management"""
    print("🔑 Testing API Key Management System")
    
    credentials_path = Path("/Users/camdouglas/quark/data/credentials/all_api_keys.json")
    key_manager = APIKeyManager(credentials_path)
    
    # Check all keys
    statuses = await key_manager.check_all_keys()
    
    # Generate report
    report = key_manager.get_key_report()
    
    print("\n📊 API Key Report:")
    print(f"   Total keys: {report['summary']['total_keys']}")
    print(f"   Active: {report['summary']['active_keys']}")
    print(f"   Expired: {report['summary']['expired_keys']}")
    print(f"   Invalid: {report['summary']['invalid_keys']}")
    
    # Show expiring keys
    if report['expiring_soon']:
        print(f"\n⏰ Keys expiring soon:")
        for service, expiry in report['expiring_soon']:
            days = (expiry - datetime.now()).days
            print(f"   • {service}: {days} days")
    
    # Show renewal instructions
    if report['renewal_instructions']:
        print(f"\n🔄 Renewal instructions available for:")
        for service in report['renewal_instructions']:
            print(f"   • {service}")
    
    print("\n✅ API Key management test complete")

if __name__ == "__main__":
    asyncio.run(main())
