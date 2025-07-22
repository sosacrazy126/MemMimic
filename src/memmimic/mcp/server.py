#!/usr/bin/env python3
"""
MemMimic MCP Server
Python-based MCP server providing all 13 MemMimic tools to Claude Desktop
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from mcp.server.models import InitializeParams, ServerCapabilities, TextContent
    from mcp.server.server import NotificationOptions, Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool
except ImportError as e:
    print(f"❌ Error importing MCP dependencies: {e}", file=sys.stderr)
    print("❌ Please install: pip install mcp", file=sys.stderr)
    sys.exit(1)

try:
    from memmimic.api import create_memmimic
    from memmimic.errors import get_error_logger
except ImportError as e:
    print(f"❌ Error importing MemMimic: {e}", file=sys.stderr)
    sys.exit(1)


class MemMimicMCPServer:
    """MemMimic MCP Server providing all 13 tools"""
    
    def __init__(self):
        self.server = Server("memmimic")
        self.logger = get_error_logger("mcp_server")
        self.memmimic_api = None
        
        # Initialize MemMimic API
        try:
            self.memmimic_api = create_memmimic()
            self.logger.info("MemMimic API initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MemMimic API: {e}")
            raise
        
        self._register_tools()
        self._register_handlers()

    def _register_tools(self):
        """Register all 13 MemMimic tools"""
        tools = [
            Tool(
                name="memmimic_remember",
                description="Store a memory with automatic CXD classification and importance scoring",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to remember"},
                        "memory_type": {"type": "string", "description": "Type/category of memory", "default": "general"},
                        "importance": {"type": "number", "description": "Importance score (0.0-1.0)", "default": 0.5}
                    },
                    "required": ["content"]
                }
            ),
            Tool(
                name="memmimic_recall_cxd", 
                description="Search memories using hybrid semantic + WordNet search with CXD filtering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "function_filter": {"type": "string", "description": "CXD function filter", "enum": ["ALL", "CONTROL", "CONTEXT", "DATA"], "default": "ALL"},
                        "limit": {"type": "integer", "description": "Max results to return", "default": 5}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="memmimic_status",
                description="Get comprehensive system status including health, statistics, and guidance",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="memmimic_think",
                description="Socratic reasoning and analysis using memory context",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to analyze"},
                        "depth": {"type": "integer", "description": "Analysis depth (1-5)", "default": 3}
                    },
                    "required": ["topic"]
                }
            ),
            Tool(
                name="memmimic_save_tale",
                description="Save narrative/story with automatic versioning and conflict detection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tale_name": {"type": "string", "description": "Unique tale identifier"},
                        "content": {"type": "string", "description": "Tale content"},
                        "category": {"type": "string", "description": "Tale category", "default": "misc"}
                    },
                    "required": ["tale_name", "content"]
                }
            ),
            Tool(
                name="memmimic_load_tale",
                description="Load a specific tale by name",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tale_name": {"type": "string", "description": "Tale name to load"}
                    },
                    "required": ["tale_name"]
                }
            ),
            Tool(
                name="memmimic_tales",
                description="List all available tales with metadata",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="memmimic_context_tale",
                description="Generate contextual narrative from related memories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic for context generation"},
                        "max_memories": {"type": "integer", "description": "Max memories to include", "default": 10}
                    },
                    "required": ["topic"]
                }
            ),
            Tool(
                name="memmimic_delete_tale",
                description="Delete a tale permanently",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tale_name": {"type": "string", "description": "Tale name to delete"}
                    },
                    "required": ["tale_name"]
                }
            ),
            Tool(
                name="memmimic_update_memory_guided",
                description="Update existing memory with guided validation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID to update"},
                        "new_content": {"type": "string", "description": "New memory content"},
                        "reason": {"type": "string", "description": "Reason for update"}
                    },
                    "required": ["memory_id", "new_content", "reason"]
                }
            ),
            Tool(
                name="memmimic_delete_memory_guided",
                description="Delete memory with safety checks and confirmation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID to delete"},
                        "reason": {"type": "string", "description": "Reason for deletion"}
                    },
                    "required": ["memory_id", "reason"]
                }
            ),
            Tool(
                name="memmimic_analyze_patterns",
                description="Analyze patterns in memory data with statistical insights",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string", "description": "Type of analysis", "default": "comprehensive"}
                    }
                }
            ),
            Tool(
                name="memmimic_remember_with_quality",
                description="Enhanced memory storage with quality validation and deduplication",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to remember"},
                        "memory_type": {"type": "string", "description": "Memory type/category", "default": "general"},
                        "quality_threshold": {"type": "number", "description": "Quality threshold (0.0-1.0)", "default": 0.7}
                    },
                    "required": ["content"]
                }
            )
        ]
        
        for tool in tools:
            self.server.list_tools.append(tool)
            
    def _register_handlers(self):
        """Register MCP handlers"""
        
        @self.server.call_tool()
        async def handle_tool_call(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls by routing to appropriate MemMimic functions"""
            try:
                if name == "memmimic_remember":
                    result = await self.memmimic_api.remember_async(
                        arguments["content"], 
                        arguments.get("memory_type", "general"),
                        importance=arguments.get("importance", 0.5)
                    )
                elif name == "memmimic_recall_cxd":
                    result = await self.memmimic_api.recall_cxd_async(
                        arguments["query"],
                        function_filter=arguments.get("function_filter", "ALL"),
                        limit=arguments.get("limit", 5)
                    )
                elif name == "memmimic_status":
                    result = await self.memmimic_api.get_status_async()
                elif name == "memmimic_think":
                    result = await self.memmimic_api.socratic_dialogue(
                        arguments["topic"],
                        depth=arguments.get("depth", 3)
                    )
                elif name == "memmimic_save_tale":
                    result = await self.memmimic_api.save_tale_async(
                        arguments["tale_name"],
                        arguments["content"], 
                        category=arguments.get("category", "misc")
                    )
                elif name == "memmimic_load_tale":
                    result = await self.memmimic_api.load_tale_async(arguments["tale_name"])
                elif name == "memmimic_tales":
                    result = await self.memmimic_api.list_tales_async()
                elif name == "memmimic_context_tale":
                    result = await self.memmimic_api.context_tale_async(
                        arguments["topic"],
                        max_memories=arguments.get("max_memories", 10)
                    )
                elif name == "memmimic_delete_tale":
                    result = await self.memmimic_api.delete_tale_async(arguments["tale_name"])
                elif name == "memmimic_update_memory_guided":
                    result = await self.memmimic_api.update_memory_guided_async(
                        arguments["memory_id"],
                        arguments["new_content"],
                        arguments["reason"]
                    )
                elif name == "memmimic_delete_memory_guided":
                    result = await self.memmimic_api.delete_memory_guided_async(
                        arguments["memory_id"],
                        arguments["reason"]
                    )
                elif name == "memmimic_analyze_patterns":
                    result = await self.memmimic_api.analyze_patterns(
                        analysis_type=arguments.get("analysis_type", "comprehensive")
                    )
                elif name == "memmimic_remember_with_quality":
                    result = await self.memmimic_api.remember_with_quality_async(
                        arguments["content"],
                        memory_type=arguments.get("memory_type", "general"),
                        quality_threshold=arguments.get("quality_threshold", 0.7)
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                # Format result as string if it's a dict/object
                if isinstance(result, dict):
                    result = str(result)
                elif not isinstance(result, str):
                    result = str(result)
                    
                return [TextContent(type="text", text=result)]
                
            except Exception as e:
                error_msg = f"Error executing {name}: {str(e)}"
                self.logger.error(error_msg)
                return [TextContent(type="text", text=f"❌ {error_msg}")]

    async def run(self):
        """Run the MCP server"""
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializeParams(
                        protocolVersion="2024-11-05",
                        capabilities={},
                        clientInfo={"name": "memmimic", "version": "1.0.0"}
                    )
                )
        except Exception as e:
            self.logger.error(f"MCP server error: {e}")
            raise


async def main():
    """Main entry point"""
    try:
        server = MemMimicMCPServer()
        await server.run()
    except KeyboardInterrupt:
        print("✅ MemMimic MCP server stopped", file=sys.stderr)
    except Exception as e:
        print(f"❌ MemMimic MCP server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())