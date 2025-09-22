#!/usr/bin/env python3
"""
CDX Server API Integration for Quark
=====================================
This module provides integration with the CDX Server API (pywb) for
querying web archive capture indexes.

CDX (Capture/Crawl inDeX) is a standard format for web archive indexes.
The CDX Server API allows programmatic access to query these indexes.

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, urlencode

import requests
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    CDX_CONFIG = credentials['services']['cdx_server']


class CDXServerClient:
    """Client for interacting with CDX Server API."""
    
    def __init__(self, base_url: str = "http://localhost:8080", collection: str = "pywb"):
        """
        Initialize CDX Server client.
        
        Args:
            base_url: Base URL for the CDX server
            collection: Collection name to query
        """
        self.base_url = base_url.rstrip('/')
        self.collection = collection
        self.cdx_endpoint = f"{self.base_url}/{collection}-cdx"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quark-CDX-Integration/1.0'
        })
    
    def search(
        self,
        url: str,
        match_type: str = 'exact',
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        output: str = 'json',
        filters: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> Union[Dict, List, str]:
        """
        Search CDX index for URL captures.
        
        Args:
            url: URL to search for
            match_type: 'exact', 'prefix', 'host', 'domain'
            from_timestamp: Start timestamp (inclusive)
            to_timestamp: End timestamp (inclusive)
            limit: Maximum number of results
            sort: 'reverse' or 'closest' (requires closest param)
            output: 'json' or 'text'
            filters: List of field filters
            fields: Fields to include in output
            page: Page number for pagination
            page_size: Results per page
            
        Returns:
            Search results in requested format
        """
        params = {'url': url}
        
        # Add match type
        if match_type != 'exact':
            params['matchType'] = match_type
        
        # Add time range
        if from_timestamp:
            params['from'] = from_timestamp
        if to_timestamp:
            params['to'] = to_timestamp
        
        # Add limit
        if limit:
            params['limit'] = limit
        
        # Add sort
        if sort:
            params['sort'] = sort
        
        # Add output format
        params['output'] = output
        
        # Add filters
        if filters:
            for f in filters:
                if 'filter' not in params:
                    params['filter'] = []
                if isinstance(params['filter'], str):
                    params['filter'] = [params['filter']]
                params['filter'].append(f)
        
        # Add field selection
        if fields:
            params['fl'] = ','.join(fields)
        
        # Add pagination
        if page is not None:
            params['page'] = page
        if page_size:
            params['pageSize'] = page_size
        
        logger.info(f"Searching CDX for: {url}")
        
        try:
            response = self.session.get(self.cdx_endpoint, params=params)
            response.raise_for_status()
            
            if output == 'json':
                return response.json()
            else:
                return response.text
                
        except requests.exceptions.RequestException as e:
            logger.error(f"CDX search error: {e}")
            raise
    
    def search_prefix(self, url_prefix: str, **kwargs) -> Union[Dict, List, str]:
        """
        Search for all URLs with a given prefix.
        
        Args:
            url_prefix: URL prefix to match
            **kwargs: Additional search parameters
            
        Returns:
            Search results
        """
        return self.search(url_prefix, match_type='prefix', **kwargs)
    
    def search_domain(self, domain: str, **kwargs) -> Union[Dict, List, str]:
        """
        Search for all URLs in a domain (including subdomains).
        
        Args:
            domain: Domain to search
            **kwargs: Additional search parameters
            
        Returns:
            Search results
        """
        return self.search(domain, match_type='domain', **kwargs)
    
    def get_page_count(self, url: str, **kwargs) -> Dict[str, int]:
        """
        Get pagination information for a query.
        
        Args:
            url: URL to search for
            **kwargs: Additional search parameters
            
        Returns:
            Dict with 'pages', 'blocks', 'pageSize'
        """
        kwargs['showNumPages'] = 'true'
        kwargs.pop('page', None)  # Remove page param
        kwargs.pop('limit', None)  # Remove limit param
        
        params = {'url': url, **kwargs}
        
        response = self.session.get(self.cdx_endpoint, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def filter_by_status(self, url: str, status: int, **kwargs) -> Union[Dict, List, str]:
        """
        Search for captures with specific HTTP status.
        
        Args:
            url: URL to search
            status: HTTP status code
            **kwargs: Additional parameters
            
        Returns:
            Filtered results
        """
        filters = kwargs.get('filters', [])
        filters.append(f'=status:{status}')
        kwargs['filters'] = filters
        
        return self.search(url, **kwargs)
    
    def filter_by_mime(self, url: str, mime_type: str, **kwargs) -> Union[Dict, List, str]:
        """
        Search for captures with specific MIME type.
        
        Args:
            url: URL to search
            mime_type: MIME type to filter
            **kwargs: Additional parameters
            
        Returns:
            Filtered results
        """
        filters = kwargs.get('filters', [])
        filters.append(f'mime:{mime_type}')
        kwargs['filters'] = filters
        
        return self.search(url, **kwargs)
    
    def get_closest(self, url: str, timestamp: str, **kwargs) -> Union[Dict, List, str]:
        """
        Find captures closest to a specific timestamp.
        
        Args:
            url: URL to search
            timestamp: Target timestamp
            **kwargs: Additional parameters
            
        Returns:
            Closest captures
        """
        kwargs['sort'] = 'closest'
        kwargs['closest'] = timestamp
        
        return self.search(url, **kwargs)
    
    def is_server_running(self) -> bool:
        """
        Check if CDX server is running.
        
        Returns:
            True if server is accessible
        """
        try:
            response = self.session.get(f"{self.base_url}")
            return response.status_code < 500
        except requests.exceptions.RequestException:
            return False


class PyWBManager:
    """Manager for pywb installation and configuration."""
    
    @staticmethod
    def is_installed() -> bool:
        """Check if pywb is installed."""
        try:
            import pywb
            return True
        except ImportError:
            return False
    
    @staticmethod
    def install():
        """Install pywb package."""
        logger.info("Installing pywb...")
        subprocess.run(["pip", "install", "pywb"], check=True)
    
    @staticmethod
    def init_collection(collection_name: str = "quark", directory: str = "."):
        """
        Initialize a new pywb collection.
        
        Args:
            collection_name: Name for the collection
            directory: Directory to create collection in
        """
        logger.info(f"Initializing collection: {collection_name}")
        subprocess.run(
            ["wb-manager", "init", collection_name],
            cwd=directory,
            check=True
        )
    
    @staticmethod
    def add_warc(collection: str, warc_path: str, directory: str = "."):
        """
        Add a WARC file to collection.
        
        Args:
            collection: Collection name
            warc_path: Path to WARC file
            directory: Collection directory
        """
        logger.info(f"Adding WARC to {collection}: {warc_path}")
        subprocess.run(
            ["wb-manager", "add", collection, warc_path],
            cwd=directory,
            check=True
        )
    
    @staticmethod
    def create_config(directory: str = ".", enable_cdx: bool = True):
        """
        Create pywb configuration file.
        
        Args:
            directory: Directory for config file
            enable_cdx: Enable CDX API endpoints
        """
        config = {
            'enable_cdx_api': enable_cdx,
            'collections': {
                'quark': {
                    'index_paths': ['indexes'],
                    'archive_paths': ['archive']
                }
            },
            'port': 8080
        }
        
        config_path = Path(directory) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Created config at: {config_path}")
        return config_path
    
    @staticmethod
    def start_server(mode: str = "wayback", directory: str = "."):
        """
        Start pywb server.
        
        Args:
            mode: 'wayback' for full mode or 'cdx-server' for CDX only
            directory: Directory with collections
            
        Returns:
            Subprocess object
        """
        logger.info(f"Starting pywb in {mode} mode...")
        process = subprocess.Popen(
            [mode],
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(2)
        
        if process.poll() is not None:
            stderr = process.stderr.read().decode() if process.stderr else ""
            raise RuntimeError(f"Failed to start server: {stderr}")
        
        logger.info(f"Server started on http://localhost:8080")
        return process


def demonstrate_cdx_queries(client: CDXServerClient):
    """
    Demonstrate CDX query capabilities.
    
    Args:
        client: CDX server client
    """
    print("\nCDX Query Examples")
    print("=" * 60)
    
    # Example 1: Search for exact URL
    print("\n1. Exact URL search")
    print("-" * 40)
    print("Query: example.com")
    print("(Would return all captures of example.com)")
    
    # Example 2: Prefix search
    print("\n2. Prefix search")
    print("-" * 40)
    print("Query: example.com/path/* with prefix match")
    print("(Would return all captures under /path/)")
    
    # Example 3: Domain search
    print("\n3. Domain search")
    print("-" * 40)
    print("Query: *.example.com")
    print("(Would return all subdomains)")
    
    # Example 4: Filter by MIME type
    print("\n4. Filter by MIME type")
    print("-" * 40)
    print("Query: example.com/* with mime:text/html")
    print("(Would return only HTML pages)")
    
    # Example 5: Time range search
    print("\n5. Time range search")
    print("-" * 40)
    print("Query: example.com from 2020 to 2021")
    print("(Would return captures from 2020-2021)")
    
    # Example 6: Status code filter
    print("\n6. HTTP status filter")
    print("-" * 40)
    print("Query: example.com/* with status:404")
    print("(Would return all 404 errors)")
    
    # Example 7: Pagination
    print("\n7. Paginated results")
    print("-" * 40)
    print("Query: example.com/* page=1, pageSize=10")
    print("(Would return page 1 with 10 results)")
    
    # Example 8: Field selection
    print("\n8. Select specific fields")
    print("-" * 40)
    print("Query: example.com with fields: url,timestamp,status")
    print("(Would return only selected fields)")
    
    # Example 9: Regular expression filter
    print("\n9. Regex filtering")
    print("-" * 40)
    print("Query: example.com/* with filter ~url:.*\\.pdf$")
    print("(Would return all PDF files)")
    
    # Example 10: Combined filters
    print("\n10. Combined filters")
    print("-" * 40)
    print("Query: example.com/* with multiple filters:")
    print("  - mime:text/html")
    print("  - !=status:404")
    print("  - timestamp range: 2020-2021")
    print("(Complex filtering example)")


def setup_demo_archive():
    """
    Set up a demo archive for testing.
    
    Returns:
        Path to demo archive directory
    """
    # Create demo directory
    demo_dir = Path(__file__).parent.parent / "data" / "web_archives"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create collections directory
    collections_dir = demo_dir / "collections"
    collections_dir.mkdir(exist_ok=True)
    
    # Create config
    config_path = demo_dir / "config.yaml"
    if not config_path.exists():
        config = {
            'enable_cdx_api': True,
            'collections': {},
            'port': 8080
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    print(f"\nDemo archive directory created at: {demo_dir}")
    print(f"Config file: {config_path}")
    
    return demo_dir


def main():
    """Example usage and setup of CDX Server."""
    print("=" * 60)
    print("CDX Server API Integration")
    print("Quark System - Web Archive Index Access")
    print("=" * 60)
    
    # Check if pywb is installed
    print("\n1. Checking pywb installation...")
    if not PyWBManager.is_installed():
        print("  pywb not installed.")
        print("  To install: pip install pywb")
        print("\n  Installation command:")
        print("    pip install pywb")
    else:
        print("  ✓ pywb is installed")
    
    # Set up demo archive structure
    print("\n2. Setting up demo archive structure...")
    demo_dir = setup_demo_archive()
    print("  ✓ Archive structure created")
    
    # Initialize CDX client (even if server not running)
    client = CDXServerClient()
    
    # Check if server is running
    print("\n3. Checking CDX server status...")
    if client.is_server_running():
        print("  ✓ CDX server is running")
        
        # Try a test query
        print("\n4. Testing CDX query...")
        try:
            # This would work if there were actual archives
            result = client.search(
                'example.com',
                output='json',
                limit=5
            )
            print(f"  Query successful: {len(result)} results")
        except Exception as e:
            print(f"  Query failed (expected if no archives): {e}")
    else:
        print("  ✗ CDX server not running")
        print("\n  To start the server:")
        print("    1. Navigate to archive directory:")
        print(f"       cd {demo_dir}")
        print("    2. Initialize a collection:")
        print("       wb-manager init quark")
        print("    3. Add WARC files (if you have any):")
        print("       wb-manager add quark /path/to/file.warc.gz")
        print("    4. Start the server:")
        print("       wayback")
        print("    5. Access CDX API at:")
        print("       http://localhost:8080/quark-cdx")
    
    # Show example queries
    print("\n5. CDX Query Examples...")
    demonstrate_cdx_queries(client)
    
    # Create setup script
    print("\n6. Creating setup script...")
    setup_script = demo_dir / "setup_cdx_server.sh"
    
    with open(setup_script, 'w') as f:
        f.write("""#!/bin/bash
# CDX Server Setup Script for Quark

echo "Setting up CDX Server for Quark..."

# Install pywb if not installed
if ! python -c "import pywb" 2>/dev/null; then
    echo "Installing pywb..."
    pip install pywb
fi

# Initialize collection if not exists
if [ ! -d "collections/quark" ]; then
    echo "Initializing quark collection..."
    wb-manager init quark
fi

# Start server
echo "Starting CDX server on http://localhost:8080"
echo "CDX API will be available at: http://localhost:8080/quark-cdx"
wayback
""")
    
    setup_script.chmod(0o755)
    print(f"  Created: {setup_script}")
    
    # Show example Python usage
    print("\n7. Python Usage Example:")
    print("-" * 40)
    print("""
from tools_utilities.cdx_server_integration import CDXServerClient

# Initialize client
client = CDXServerClient(collection='quark')

# Search for exact URL
results = client.search('https://example.com')

# Search with filters
html_pages = client.filter_by_mime(
    'https://example.com/*',
    'text/html',
    limit=100
)

# Search domain and subdomains
domain_captures = client.search_domain(
    'example.com',
    from_timestamp='2020',
    to_timestamp='2024'
)

# Get closest capture to timestamp
closest = client.get_closest(
    'https://example.com',
    '20230615120000'
)

# Paginated search
page_info = client.get_page_count('https://example.com/*')
print(f"Total pages: {page_info['pages']}")

for page in range(page_info['pages']):
    results = client.search(
        'https://example.com/*',
        page=page,
        pageSize=10
    )
    # Process results...
""")
    
    print("\n" + "=" * 60)
    print("CDX Server API configuration complete!")
    print("Note: This is a self-hosted service.")
    print(f"Setup script: {setup_script}")
    print("=" * 60)


if __name__ == "__main__":
    main()

