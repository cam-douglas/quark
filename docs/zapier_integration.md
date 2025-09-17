# MCP Integrations – Zapier, Context7 & Other Services

> Status: draft (2025-09-02)

This document explains the MCP (Model Context Protocol) integrations available in Quark, including Zapier and Context7 for enhanced AI coding assistance.

---

## 1. Hosted Zapier MCP (recommended for most users)

Zapier now hosts an MCP server for you – no code, no runtime.

### Setup

1. Visit <https://mcp.zapier.com>.
2. **+ New MCP Server** → choose **Cursor** → name it → **Create**.
3. Add tools (Slack → *Send Channel Message*, Sheets → *Create Row*, etc.).
4. Copy the server URL (looks like `https://mcp.zapier.com/mcp/<id>`).
5. Append to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "Zapier MCP": {
      "httpUrl": "https://mcp.zapier.com/mcp/<id>",
      "timeout": 180000
    }
  }
}
```

6. Reload Cursor → ask *“What Zapier tools do I have?”*.

### Pros / Cons

|                           | Hosted MCP |
|---------------------------|------------|
| Maintenance               | Zero (Zapier updates server) |
| Availability              | Always-on, laptop may be closed |
| Custom code               | ❌ Not supported |
| Usage limits              | Zapier plan (actions billed) |
| Data residency            | Zapier cloud |

---

## 2. Local Custom Server (advanced / on-prem)

We scaffolded a project in `~/zapier-my-app/` using the Zapier Platform CLI.

### Local steps already done

```
npm -g install zapier-platform-cli
zapier init --template minimal .
zapier login   # OAuth via browser
```

`zapier test` now passes via TS-Node wrapper.

### Exposing as MCP

1. Add FastMCP (or any MCP lib) to the project:
   ```bash
   npm i mcp-server-fastmcp
   ```
2. Create `mcp-server.ts` that wraps your Zapier actions and runs `mcp.run()`.
3. Add to `~/.cursor/mcp.json`:
   ```json
   "Zapier Local": {
     "command": "node",
     "args": ["mcp-server.js"]
   }
   ```

### Pros / Cons

|                           | Local server |
|---------------------------|--------------|
| Maintenance               | You own updates |
| Availability              | Must keep process running |
| Custom logic              | ✅ Full freedom (data shaping, chaining) |
| Usage limits              | No Zapier billing; only downstream APIs |
| Data residency            | Stays on-prem |

---

## Choosing

* **Hosted** – fastest way to let Cursor automate Slack, Gmail, Sheets, etc.
* **Local** – when you need bespoke business logic or must keep credentials internal.

---

## 3. Context7 MCP (Code Documentation)

Context7 provides up-to-date code documentation for LLMs and AI code editors.

### Setup

Context7 is already configured in this project:

1. **MCP Server**: Added to `~/.cursor/mcp.json`:
   ```json
   "context7": {
     "command": "npx",
     "args": [
       "-y",
       "@upstash/context7-mcp",
       "--api-key",
       "ctx7sk-d41fc2b6-2bb9-4dca-8585-36d25e769266"
     ]
   }
   ```

2. **API Key**: Stored securely in `/Users/camdouglas/quark/data/credentials/all_api_keys.json`

3. **Cursor Rule**: Automatically invokes Context7 for code documentation requests via `.cursor/rules/context7-mcp-integration.mdc`

### Usage

Context7 automatically activates when you:
- Request code examples or implementation patterns
- Ask for setup or configuration steps for libraries/frameworks  
- Need API documentation or library usage information
- Want to understand how to use a specific library or framework
- Ask about best practices for specific technologies

### Available Tools

- `resolve-library-id`: Resolves library names to Context7-compatible IDs
- `get-library-docs`: Fetches up-to-date documentation for libraries

---

## Next Steps

* Add real triggers/actions in `src/` for Zapier (currently only template code exists).
* Optionally containerise local servers for team deployment.
* Monitor Zapier's beta limits → <https://docs.zapier.com/mcp/>.
* Test Context7 integration with various library documentation requests.
