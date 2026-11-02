"""Small MCP client for connecting to the local documentation MCP server."""
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server = StdioServerParameters(command="python", args=["-m", "mcp_layer.server"])
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = await session.list_tools()
        print("Connected to Documentation MCP Server")
        for tool in tools.tools:
            print(f"- {tool.name}: {tool.description}")
        while True:
            name = input("\nTool name (or exit): ").strip()
            if name.lower() in {"exit", "quit"}:
                break
            arguments = {}
            if name == "search_documentation":
                arguments = {"query": input("Query: ").strip(), "limit": 5}
            elif name == "discover_documentation":
                arguments = {"technology": input("Technology: ").strip(), "topic": input("Topic: ").strip(), "max_results": 5}
            elif name == "prepare_documentation":
                arguments = {"url": input("Official documentation URL: ").strip()}
            elif name == "list_pending_documentation":
                arguments = {}
            elif name == "ingest_approved_documentation":
                arguments = {"proposal_id": input("Approved proposal ID: ").strip()}
            else:
                print("Unknown tool")
                continue
            result = await session.call_tool(name, arguments)
            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
