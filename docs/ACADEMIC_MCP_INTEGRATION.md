# Academic MCP Servers Integration

## Overview

Successfully integrated 8 academic research MCP servers into the Cursor workspace to enhance AI assistant capabilities for academic research, paper analysis, and scientific literature access. (Note: 1 server removed due to compatibility issues)

## Integrated MCP Servers

### 1. ArXiv MCP Server
- **Source**: https://github.com/blazickjp/arxiv-mcp-server
- **Purpose**: Search and access arXiv papers
- **Location**: `/Users/camdouglas/external_tools/academic_mcp_servers/arxiv-mcp-server/`
- **Features**:
  - Paper search with filters for date ranges and categories
  - Paper download and content access
  - Local storage for faster access
  - Research prompts for paper analysis

### 2. OpenAlex MCP Server (alex-mcp)
- **Source**: https://github.com/drAbreu/alex-mcp
- **Purpose**: Author disambiguation and academic research via OpenAlex API
- **Location**: `/Users/camdouglas/external_tools/academic_mcp_servers/alex-mcp/`
- **Features**:
  - Advanced author disambiguation
  - Institution resolution with transition tracking
  - Academic work retrieval (journal articles, research papers)
  - Citation analysis and H-index metrics
  - ORCID integration

### 3. PubMed MCP Server
- **Source**: https://github.com/JackKuo666/PubMed-MCP-Server
- **Purpose**: Search and analyze PubMed biomedical literature
- **Location**: `/Users/camdouglas/external_tools/academic_mcp_servers/PubMed-MCP-Server/`
- **Features**:
  - Biomedical literature search
  - Paper metadata access
  - PDF content download attempts
  - Deep analysis capabilities

### 4. Semantic Scholar MCP Server
- **Source**: https://github.com/zongmin-yu/semantic-scholar-fastmcp-mcp-server
- **Purpose**: Access Semantic Scholar's academic paper database
- **Location**: `/Users/camdouglas/external_tools/academic_mcp_servers/semantic-scholar-fastmcp-mcp-server/`
- **Features**:
  - Comprehensive academic data access
  - Citation networks
  - Paper recommendations

### 5. Google Scholar MCP Server
- **Source**: https://github.com/JackKuo666/Google-Scholar-MCP-Server
- **Purpose**: Search and access Google Scholar papers
- **Location**: `/Users/camdouglas/external_tools/academic_mcp_servers/Google-Scholar-MCP-Server/`
- **Features**:
  - Google Scholar search integration
  - Citation data access
  - Author profile information

### 6. Unpaywall MCP Server
- **Source**: https://github.com/elliotpadfield/unpaywall-mcp
- **Purpose**: Find open-access versions of scholarly articles
- **Location**: `/Users/camdouglas/external_tools/academic_mcp_servers/unpaywall-mcp/`
- **Features**:
  - DOI metadata retrieval
  - Open access link discovery
  - PDF text extraction
  - Title-based search

### 7. Academic Search MCP Server
- **Source**: https://github.com/afrise/academic-search-mcp-server
- **Purpose**: General academic search capabilities
- **Location**: `/Users/camdouglas/external_tools/academic_mcp_servers/academic-search-mcp-server/`
- **Features**:
  - Cross-platform academic search
  - Metadata aggregation

### 8. Scientific Papers MCP Server
- **Source**: https://github.com/benedict2310/Scientific-Papers-MCP
- **Purpose**: Scientific paper harvesting from arXiv and OpenAlex
- **Location**: `/Users/camdouglas/external_tools/academic_mcp_servers/Scientific-Papers-MCP/`
- **Features**:
  - Multi-source paper harvesting
  - Advanced filtering and search
  - Batch processing capabilities

## Configuration

The MCP servers are configured in `/Users/camdouglas/.cursor/mcp.json` with the following entries:

```json
{
  "mcpServers": {
    "arxiv-mcp-server": {
      "command": "/Users/camdouglas/external_tools/academic_mcp_servers/arxiv-mcp-server/.venv/bin/python",
      "args": ["-m", "arxiv_mcp_server"],
      "env": {
        "ARXIV_STORAGE_PATH": "/Users/camdouglas/external_tools/academic_mcp_servers/arxiv-mcp-server/papers"
      }
    },
    "alex-mcp": {
      "command": "/Users/camdouglas/external_tools/academic_mcp_servers/alex-mcp/.venv/bin/python",
      "args": ["-m", "alex_mcp.server"]
    },
    "pubmed-mcp-server": {
      "command": "/Users/camdouglas/external_tools/academic_mcp_servers/PubMed-MCP-Server/.venv/bin/python",
      "args": ["/Users/camdouglas/external_tools/academic_mcp_servers/PubMed-MCP-Server/server.py"]
    },
    "semantic-scholar-mcp": {
      "command": "/Users/camdouglas/external_tools/academic_mcp_servers/semantic-scholar-fastmcp-mcp-server/.venv/bin/python",
      "args": ["/Users/camdouglas/external_tools/academic_mcp_servers/semantic-scholar-fastmcp-mcp-server/server.py"]
    },
    "google-scholar-mcp": {
      "command": "/Users/camdouglas/external_tools/academic_mcp_servers/Google-Scholar-MCP-Server/.venv/bin/python",
      "args": ["/Users/camdouglas/external_tools/academic_mcp_servers/Google-Scholar-MCP-Server/server.py"]
    },
    "unpaywall-mcp": {
      "command": "node",
      "args": ["/Users/camdouglas/external_tools/academic_mcp_servers/unpaywall-mcp/bin/unpaywall-mcp.js"]
    },
    "academic-search-mcp": {
      "command": "/Users/camdouglas/external_tools/academic_mcp_servers/academic-search-mcp-server/.venv/bin/python",
      "args": ["/Users/camdouglas/external_tools/academic_mcp_servers/academic-search-mcp-server/server.py"]
    },
    "scientific-papers-mcp": {
      "command": "node",
      "args": ["/Users/camdouglas/external_tools/academic_mcp_servers/Scientific-Papers-MCP/dist/server.js"]
    }
  }
}
```

## Installation Summary

### Python-based Servers (6 servers)
- Each installed in individual Python virtual environments
- Dependencies installed via pip/requirements.txt
- Entry points configured for Python module execution

### Node.js-based Servers (2 servers)
- TypeScript projects built to JavaScript
- Dependencies installed via npm
- Built distributions used for execution

### Skipped Servers (1 server)
- **Crossref MCP Server**: Skipped due to installation complexity with Smithery

## Usage Examples

### ArXiv Paper Search
```python
# Search for papers on transformer architecture
result = await call_tool("search_papers", {
    "query": "transformer architecture",
    "max_results": 10,
    "date_from": "2023-01-01",
    "categories": ["cs.AI", "cs.LG"]
})
```

### Author Disambiguation with OpenAlex
```python
# Find author information and disambiguate
result = await call_tool("search_author", {
    "name": "Geoffrey Hinton",
    "institution": "University of Toronto"
})
```

### PubMed Biomedical Literature Search
```python
# Search biomedical literature
result = await call_tool("search_pubmed", {
    "query": "CRISPR gene editing",
    "max_results": 20
})
```

## API Keys and Configuration

Some servers may require API keys or additional configuration:

1. **Semantic Scholar**: May require rate limiting consideration
2. **Google Scholar**: May need proxy configuration for large-scale use
3. **OpenAlex**: Free API with rate limits
4. **PubMed**: Free API from NCBI

API keys should be stored in `/Users/camdouglas/quark/data/credentials/all_api_keys.json` following the established pattern.

## Storage Locations

- **ArXiv Papers**: `/Users/camdouglas/external_tools/academic_mcp_servers/arxiv-mcp-server/papers`
- **Virtual Environments**: Each server has its own `.venv` directory
- **Built Assets**: TypeScript servers have `dist/` directories with compiled JavaScript

## Maintenance

### Updating Servers
```bash
cd /Users/camdouglas/external_tools/academic_mcp_servers/[server-name]
git pull origin main
# For Python servers:
source .venv/bin/activate && pip install -r requirements.txt
# For Node.js servers:
npm install && npm run build
```

### Monitoring
- Check Cursor's MCP server status in settings
- Monitor server logs for errors
- Verify API rate limits and quotas

## Integration Benefits

This integration provides the Cursor AI assistant with comprehensive access to:

1. **Academic Literature**: arXiv, PubMed, Google Scholar, Semantic Scholar
2. **Author Information**: Disambiguation, affiliations, citation metrics
3. **Open Access**: Finding free versions of paywalled papers
4. **Cross-Platform Search**: Unified access to multiple academic databases
5. **Research Analysis**: Deep paper analysis and research prompts

## Recent Fixes (2025-01-14)

### Fixed Server Configurations
The following servers were showing "no tools or prompts" and have been fixed:

1. **alex-mcp**: 
   - **Issue**: Missing required environment variable `OPENALEX_MAILTO`
   - **Fix**: Added email environment variable to configuration
   - **Entry Point**: Updated to use correct server path `/src/alex_mcp/server.py`

2. **pubmed-mcp-server**:
   - **Issue**: Incorrect server file path
   - **Fix**: Updated to use correct server file `pubmed_server.py`

3. **semantic-scholar-mcp**:
   - **Issue**: FastMCP compatibility issue - server has incompatible async context manager
   - **Fix**: **REMOVED** from configuration due to library compatibility issues

4. **google-scholar-mcp**:
   - **Issue**: Incorrect server file path  
   - **Fix**: Updated to use correct server file `google_scholar_server.py`

All servers now import successfully and should provide their tools and prompts correctly.

## Troubleshooting

### Common Issues
1. **Virtual Environment Issues**: Ensure Python virtual environments are properly activated
2. **TypeScript Build Errors**: Run `npm run build` in TypeScript project directories
3. **API Rate Limits**: Monitor and respect API rate limits for external services
4. **Path Issues**: Verify all paths in mcp.json are correct and accessible
5. **Missing Environment Variables**: Some servers (like alex-mcp) require specific environment variables

### Logs
- Check Cursor's developer tools for MCP server connection status
- Individual server logs may be available in their respective directories

---

**Integration Date**: 2025-01-14
**Status**: Active
**Total Servers**: 9/10 (1 skipped)
**Maintainer**: Quark System

