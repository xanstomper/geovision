#!/bin/bash
# GeoVision Launcher - sets env and starts MCP server
export NODE_OPTIONS="--max-old-space-size=4096"
cd "$(dirname "$0")"
echo "GeoVision starting..."
exec node src/mcp-server.js "$@"
