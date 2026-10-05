import asyncio
import json

from ollama import chat
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MODEL = "ministral-3:8b"


server_params = StdioServerParameters(
    command=r"C:\Users\adith\AppData\Local\Programs\Python\Python312\python.exe",
    args=[
        r"D:\Gmail-MCP-Local\mcp_gmail.py"
    ],
)


async def main():

    print("Starting Gmail MCP server...")

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            print("Initializing MCP connection...")

            await session.initialize()

            # Get available tools from the Gmail MCP server
            tools_result = await session.list_tools()

            print("\nAvailable Gmail MCP tools:\n")

            for tool in tools_result.tools:
                print(f"- {tool.name}")
                print(f"  {tool.description}")
                print()

            # Convert MCP tools to Ollama tool format
            ollama_tools = []

            for tool in tools_result.tools:

                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    }
                })

            print("----------------------------------------")
            print("Local Gmail AI is ready.")
            print(f"Model: {MODEL}")
            print("Type 'exit' to quit.")
            print("----------------------------------------\n")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a local AI assistant with access to Gmail "
                        "through MCP tools.\n\n"
                        "Use the Gmail MCP tools whenever the user asks "
                        "about Gmail or emails.\n"
                        "Never invent email information.\n"
                        "Always use the appropriate tool when actual "
                        "Gmail data is required."
                    )
                }
            ]

            while True:

                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    break

                messages.append({
                    "role": "user",
                    "content": user_input
                })

                # Ask the local Qwen model
                response = chat(
                    model=MODEL,
                    messages=messages,
                    tools=ollama_tools
                )

                assistant_message = response.message

                # Save assistant response
                messages.append(assistant_message)

                # Check whether Qwen requested any tools
                if assistant_message.tool_calls:

                    for tool_call in assistant_message.tool_calls:

                        tool_name = tool_call.function.name
                        tool_arguments = tool_call.function.arguments

                        print(
                            f"\n[MCP TOOL CALL] {tool_name}"
                        )

                        print(
                            "[Arguments]"
                        )

                        print(
                            json.dumps(
                                tool_arguments,
                                indent=2
                            )
                        )

                        try:

                            # Execute the MCP tool
                            result = await session.call_tool(
                                tool_name,
                                arguments=tool_arguments
                            )

                            result_text = ""

                            for content in result.content:

                                if hasattr(content, "text"):
                                    result_text += content.text

                                else:
                                    result_text += str(content)

                            print(
                                "[MCP TOOL RESULT RECEIVED]"
                            )

                            # Send the MCP result back to Qwen
                            messages.append({
                                "role": "tool",
                                "content": result_text
                            })

                        except Exception as e:

                            error_message = (
                                f"Error executing MCP tool "
                                f"{tool_name}: {str(e)}"
                            )

                            print(error_message)

                            messages.append({
                                "role": "tool",
                                "content": error_message
                            })

                    # Ask Qwen to generate the final answer
                    final_response = chat(
                        model=MODEL,
                        messages=messages,
                        tools=ollama_tools
                    )

                    final_message = final_response.message

                    messages.append(final_message)

                    print(
                        f"\nAI: {final_message.content}\n"
                    )

                else:

                    print(
                        f"\nAI: {assistant_message.content}\n"
                    )


if __name__ == "__main__":
    asyncio.run(main())