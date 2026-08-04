# Publishing coalent-mcp to the official MCP Registry

How to publish (and re-publish) the Coalent MCP server to
[registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io). The registry
hosts **metadata only** — the artifact users install is the `coalent` package on PyPI, so
PyPI always comes first.

The manifest is [`server.json`](server.json) at the repo root
(schema `2025-12-11`, validated). Server name: **`io.github.vectorlink-labs/coalent`**.

## Prerequisites (once per release)

1. **`coalent 0.6.1` must be live on PyPI before publishing to the registry.** The
   registry validates the package exists and verifies ownership against it.
2. **The ownership marker must be in the PyPI README.** The registry fetches the package's
   PyPI description and requires the exact string `mcp-name: io.github.vectorlink-labs/coalent`
   in it. It is already present in [README.md](README.md) as an HTML comment
   (`<!-- mcp-name: io.github.vectorlink-labs/coalent -->` in the MCP section — PyPI
   preserves HTML comments). Do not remove it; if the sdist/wheel README ever changes,
   re-check the marker survived.
3. **A GitHub account in the `Vectorlink-Labs` org.** GitHub-based auth only permits
   publishing names under `io.github.<your-user-or-org>/`, so the login below must be an
   account with access to `Vectorlink-Labs`.

## Step 1 — install `mcp-publisher`

Windows (PowerShell):

```powershell
$arch = if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq "Arm64") { "arm64" } else { "amd64" }
Invoke-WebRequest -Uri "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_windows_$arch.tar.gz" -OutFile "mcp-publisher.tar.gz"
tar xf mcp-publisher.tar.gz mcp-publisher.exe
rm mcp-publisher.tar.gz
# Move mcp-publisher.exe to a directory on your PATH
```

macOS / Linux:

```bash
curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/').tar.gz" | tar xz mcp-publisher && sudo mv mcp-publisher /usr/local/bin/
# or: brew install mcp-publisher
```

Verify: `mcp-publisher --help` lists `init / login / logout / publish`.

## Step 2 — authenticate

```bash
mcp-publisher login github
```

This starts a GitHub device flow: open the printed URL
(`https://github.com/login/device`), enter the printed code, authorize. Expect
`✓ Successfully logged in`.

## Step 3 — publish

From the repo root (where `server.json` lives):

```bash
mcp-publisher publish
```

Expect:

```text
✓ Successfully published
✓ Server io.github.vectorlink-labs/coalent version 0.6.1
```

## Step 4 — verify

```bash
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.vectorlink-labs/coalent"
```

The response JSON should contain `"name":"io.github.vectorlink-labs/coalent"` with
version `0.6.1`.

## Re-publishing a new version

For every future release: bump **both** `version` fields in `server.json` (the top-level
server version and `packages[0].version` — they track the PyPI release), publish to PyPI
first, then repeat steps 2–4. Versions are immutable in the registry; a re-publish of an
existing version is rejected.

## Troubleshooting

| Error | Fix |
| --- | --- |
| "Registry validation failed for package" | The `mcp-name: io.github.vectorlink-labs/coalent` marker is missing from the **live PyPI** README (publish the PyPI release first / check the marker survived) |
| "Invalid or expired Registry JWT token" | `mcp-publisher login github` again |
| "You do not have permission to publish this server" | The logged-in GitHub account cannot publish under `io.github.vectorlink-labs/` — log in with an account in the org |
