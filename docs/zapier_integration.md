# Zapier Integration – MCP & Local CLI

> Status: draft (2025-09-02)

This document explains the two ways Quark can talk to Zapier from Cursor and other MCP-compatible assistants.

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

## Next Steps

* Add real triggers/actions in `src/` (currently only template code exists).
* Optionally containerise the local server for team deployment.
* Monitor Zapier’s beta limits → <https://docs.zapier.com/mcp/>.
