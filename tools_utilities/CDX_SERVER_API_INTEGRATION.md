# CDX Server API Integration

**Date**: 2025-01-20  
**Status**: ✔ Configured  
**Reference**: [CDX Server API Wiki](https://github.com/webrecorder/pywb/wiki/CDX-Server-API)

## Overview

The CDX Server API integration provides Quark with the ability to query web archive indexes. CDX (Capture/Crawl inDeX) is a standard format for web archive metadata. The CDX Server, part of pywb (Python Web Archiver), enables programmatic access to search and analyze archived web content.

## Key Features

### API Capabilities
- **Self-Hosted Service**: Runs locally on your infrastructure
- **No Authentication**: Default configuration (can be secured)
- **Advanced Querying**: Complex filters and search patterns
- **JSON Output**: Structured data responses
- **Pagination Support**: Handle large result sets
- **Real-time Access**: Query live archive indexes

### Search Features
1. **URL Matching**: Exact, prefix, host, domain patterns
2. **Time Filtering**: Date/time range queries
3. **Field Filtering**: MIME type, HTTP status, etc.
4. **Regular Expressions**: Advanced pattern matching
5. **Sorting Options**: Reverse chronological, closest match
6. **Field Selection**: Choose specific output fields
7. **Pagination**: Page through large results

## Installation & Setup

### 1. Install pywb
```bash
pip install pywb
```

### 2. Initialize Archive
```bash
# Navigate to archive directory
cd /Users/camdouglas/quark/data/web_archives

# Initialize a collection
wb-manager init quark

# Add WARC files (if available)
wb-manager add quark /path/to/file.warc.gz
```

### 3. Configure CDX API
Configuration file: `/data/web_archives/config.yaml`
```yaml
enable_cdx_api: true
collections:
  quark:
    index_paths: ['indexes']
    archive_paths: ['archive']
port: 8080
```

### 4. Start Server
```bash
# Full wayback mode (replay + CDX API)
wayback

# Or CDX-only mode
cdx-server
```

### 5. Access CDX API
- Base URL: `http://localhost:8080`
- CDX Endpoint: `http://localhost:8080/quark-cdx`

## API Configuration

Configuration stored in: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

```json
"cdx_server": {
    "service": "CDX Server API (pywb)",
    "endpoints": {
        "base": "http://localhost:8080",
        "collection_cdx": "http://localhost:8080/{collection}-cdx"
    },
    "api_key": "none_required",
    "installation": {
        "package": "pywb",
        "install_command": "pip install pywb"
    }
}
```

## Integration Module

**Location**: `/Users/camdouglas/quark/tools_utilities/cdx_server_integration.py`

### Usage Examples

```python
from tools_utilities.cdx_server_integration import CDXServerClient

# Initialize client
client = CDXServerClient(collection='quark')

# Search for exact URL
results = client.search('https://example.com', output='json')

# Prefix search (all pages under path)
pages = client.search_prefix('https://example.com/blog/')

# Domain search (including subdomains)
domain_captures = client.search_domain('example.com')

# Filter by MIME type
html_only = client.filter_by_mime(
    'https://example.com/*',
    'text/html'
)

# Filter by status code
errors = client.filter_by_status(
    'https://example.com/*',
    404
)

# Time range search
historical = client.search(
    'https://example.com',
    from_timestamp='20200101',
    to_timestamp='20211231'
)

# Get closest to timestamp
closest = client.get_closest(
    'https://example.com',
    '20230615120000'
)
```

## Query Parameters

### Core Parameters
| Parameter | Description | Example |
|-----------|-------------|---------|
| `url` | URL to search (required) | `example.com` |
| `matchType` | Match strategy | `exact`, `prefix`, `host`, `domain` |
| `from` | Start timestamp | `20200101` |
| `to` | End timestamp | `20211231` |
| `limit` | Max results | `100` |
| `output` | Response format | `json` or `text` |

### Filter Operators
| Operator | Syntax | Example |
|----------|--------|---------|
| Contains | `field:value` | `mime:html` |
| Exact | `=field:value` | `=status:200` |
| Regex | `~field:pattern` | `~url:.*\.pdf$` |
| Not contains | `!field:value` | `!mime:image` |
| Not exact | `!=field:value` | `!=status:404` |
| Not regex | `!~field:pattern` | `!~url:.*\.jpg$` |

### CDX Index Fields
- `urlkey`: SURT-formatted URL key
- `timestamp`: 14-digit timestamp
- `url`: Original URL
- `mime`: MIME type
- `status`: HTTP status code
- `digest`: Content hash
- `length`: Response length
- `offset`: WARC file offset
- `filename`: WARC filename

## Advanced Queries

### Complex Filtering
```python
# HTML pages that aren't 404s from 2023
results = client.search(
    'https://example.com/*',
    from_timestamp='2023',
    to_timestamp='2023',
    filters=[
        'mime:text/html',
        '!=status:404'
    ]
)
```

### Pagination
```python
# Get total pages
page_info = client.get_page_count('https://example.com/*')
total_pages = page_info['pages']

# Iterate through pages
for page_num in range(total_pages):
    results = client.search(
        'https://example.com/*',
        page=page_num,
        page_size=50,
        output='json'
    )
    # Process results...
```

### Field Selection
```python
# Get only specific fields
minimal = client.search(
    'https://example.com',
    fields=['url', 'timestamp', 'status'],
    output='json'
)
```

## Integration with Quark

### Use Cases for Brain Research
1. **Literature Tracking**: Archive research paper websites
2. **Data Source Monitoring**: Track dataset availability
3. **API Documentation**: Preserve API docs over time
4. **Resource Availability**: Monitor tool/service uptime
5. **Research Reproducibility**: Archive experiment URLs
6. **Citation Preservation**: Keep referenced web content

### Scientific Applications
- Archive neuroscience databases
- Track brain atlas updates
- Monitor research tool changes
- Preserve computational resource docs
- Archive conference websites
- Track funding opportunity pages

## Setup Script

A setup script is available at:
```bash
/Users/camdouglas/quark/data/web_archives/setup_cdx_server.sh
```

Run it to automatically:
1. Install pywb
2. Initialize collection
3. Start CDX server

## Directory Structure

```
/data/web_archives/
├── config.yaml           # pywb configuration
├── setup_cdx_server.sh   # Setup script
└── collections/          # Archive collections
    └── quark/           # Quark collection
        ├── archive/     # WARC files
        └── indexes/     # CDX indexes
```

## Working with WARC Files

### Create WARC from URLs
```bash
# Record a website
wget --warc-file=mysite https://example.com

# Or use webrecorder tools
pip install warcio
warcit https://example.com -o mysite.warc.gz
```

### Add to Collection
```bash
wb-manager add quark mysite.warc.gz
```

### Reindex Collection
```bash
wb-manager reindex quark
```

## Performance Tips

1. **Use Pagination**: For large result sets
2. **Limit Fields**: Return only needed fields
3. **Add Indexes**: Use ZipNum for large archives
4. **Cache Results**: Store frequently accessed queries
5. **Filter Early**: Apply filters to reduce data transfer

## Troubleshooting

### Server Not Starting
```bash
# Check if port 8080 is in use
lsof -i :8080

# Use different port
wayback --port 8090
```

### No Results
```bash
# Check collection has data
ls collections/quark/indexes/

# Reindex if needed
wb-manager reindex quark
```

### Slow Queries
- Add `limit` parameter
- Use more specific URLs
- Enable ZipNum compression

## Additional Resources

### Documentation
- [pywb Documentation](https://pywb.readthedocs.io/)
- [CDX File Format](https://github.com/webrecorder/pywb/wiki/CDX-Index-Format)
- [WARC Specification](https://iipc.github.io/warc-specifications/)

### Tools
- [Webrecorder](https://webrecorder.net/) - Web archiving platform
- [warcio](https://github.com/webrecorder/warcio) - WARC library
- [Browsertrix](https://github.com/webrecorder/browsertrix) - Browser-based crawling

### Community
- [Web Archiving Slack](https://iipc.slack.com/)
- [IIPC](https://netpreserve.org/) - International Internet Preservation Consortium

## Status

✅ **Integration Configured**: CDX Server API is configured and ready. Server must be started locally to use. Archive structure created at `/data/web_archives/`.

