import asyncio
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import webbrowser

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Load environment variables
load_dotenv()

# DeepSeek model constant
DEEPSEEK_MODEL = "deepseek-chat"

class PaperReadingAssistant:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Initialize DeepSeek client
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        ) if DEEPSEEK_API_KEY else None
        # Initialize conversation history
        self.conversation_history = []
        # Track papers downloaded in this session
        self.downloaded_papers = []  # List of downloaded papers in this session
        # Track current paper being read
        self.current_paper = None
        self.current_paper_path = None

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script
        """
        path = Path(server_script_path).resolve()
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(path)],
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("已连接到智能论文阅读助手")
        print("可用工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

    async def process_query_with_deepseek(self, query: str) -> str:
        """Process a query using DeepSeek and available tools"""
        if not self.deepseek_client:
            return "DeepSeek API key not configured. Please set DEEPSEEK_API_KEY in your environment."
            
        # Add user query to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": query
        })

        # Get available tools
        response = await self.session.list_tools()
        available_tools = []
        for tool in response.tools:
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                }
            })

        # Add system message to guide the model
        system_content = "你是一个智能助手，可以根据用户的问题决定是否需要调用工具。如果需要调用工具，请使用适当的函数调用格式。你有以下工具可供使用：" + \
                        ", ".join([tool['function']['name'] for tool in available_tools]) + \
                        "。请一步一步完成用户请求。"
                        
        if self.current_paper:
            system_content += f" 当前正在处理的论文是: {self.current_paper}"

        system_message = {
            "role": "system", 
            "content": system_content
        }
        
        # Prepare messages with conversation history
        messages = [system_message] + self.conversation_history[-15:]  # Keep last 15 interactions

        # Process with tool calls loop
        final_response = ""
        
        for i in range(8):  # Increase limit for complex workflows
            # Call DeepSeek API
            try:
                response = self.deepseek_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    tools=available_tools if available_tools else None,
                    tool_choice="auto" if available_tools else None,
                    stream=False,
                    max_tokens=2000
                )
            except Exception as e:
                return f"调用DeepSeek API时出错: {str(e)}"

            message = response.choices[0].message
            
            # If no tool calls, we're done
            if not hasattr(message, 'tool_calls') or not message.tool_calls:
                if message.content:
                    final_response = message.content  # 直接赋值，而不是累加
                break
                
            # Add assistant's response to messages and conversation history
            assistant_msg = {
                "role": "assistant",
                "content": message.content if message.content else "",
                "tool_calls": message.tool_calls
            }
            messages.append(assistant_msg)
            self.conversation_history.append(assistant_msg)
            
            # Process tool calls
            tool_messages = []
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_result = f"调用工具 {function_name} 失败：参数格式错误"
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    })
                    continue
                
                # Execute tool call
                try:
                    result = await self.session.call_tool(function_name, function_args)
                    
                    # Extract text content from result
                    tool_result_text = ""
                    if isinstance(result, list) and len(result) > 0:
                        for item in result:
                            if hasattr(item, 'text'):
                                tool_result_text += item.text
                            else:
                                tool_result_text += str(item)
                    else:
                        tool_result_text = str(result)
                    
                    # Special handling for download_paper to track current paper
                    if function_name == "download_paper":
                        # Extract file path from result
                        if "已成功下载论文至:" in tool_result_text:
                            path_start = tool_result_text.find(": ") + 2
                            self.current_paper_path = tool_result_text[path_start:].strip()
                            # Extract paper filename from path
                            self.current_paper = Path(self.current_paper_path).name
                            # Add to downloaded papers list if not already there
                            if self.current_paper not in self.downloaded_papers:
                                self.downloaded_papers.append(self.current_paper)
                            # Automatically open PDF
                            try:
                                webbrowser.open(self.current_paper_path)
                                tool_result_text += "\n已自动打开PDF文件供您阅读。"
                            except Exception as e:
                                tool_result_text += f"\n无法自动打开PDF文件: {str(e)}"
                    
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result_text
                    }
                    tool_messages.append(tool_msg)
                    
                except Exception as e:
                    tool_result = f"调用工具 {function_name} 出错: {str(e)}"
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    }
                    tool_messages.append(tool_msg)
            
            # Add tool results to messages and conversation history
            messages.extend(tool_messages)
            self.conversation_history.extend(tool_messages)
            
            # Get next response from model with tool results
            try:
                next_response = self.deepseek_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    stream=False,
                    max_tokens=2000
                )
                
                next_message = next_response.choices[0].message
                if next_message.content:
                    final_response = next_message.content  # 直接赋值，而不是累加
                    
                # Update messages with next response
                next_msg = {
                    "role": "assistant",
                    "content": next_message.content
                }
                messages.append(next_msg)
                self.conversation_history.append(next_msg)
                
            except Exception as e:
                return f"调用DeepSeek API时出错: {str(e)}"
        
        # Limit conversation history to prevent it from growing too large
        if len(self.conversation_history) > 30:
            # Keep system message and last 30 interactions
            self.conversation_history = self.conversation_history[-30:]
            
        return final_response if final_response else "已完成操作。"

    async def interactive_session(self):
        """运行交互式会话"""
        print("\n=== 智能论文阅读助手 ===")
        print("您好！我是您的智能学术研究助手，由DeepSeek驱动。")
        print("您可以自由地向我提问关于学术论文的任何问题，我会根据需要调用适当的工具来帮助您。")
        print("例如，您可以问：")
        print("  - '帮我找几篇关于transformer的论文'")
        print("  - '你能分析一下Attention is All You Need这篇论文吗？'")
        print("  - '我想了解GAN的最新进展'")
        print("输入 'quit' 退出")
        
        while True:
            try:
                user_input = input("\n您: ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() == "quit":
                    # Check if there's a current paper being read
                    if self.current_paper:
                        print(f"\n您正在阅读论文 '{self.current_paper}'，是否已完成阅读？")
                        finished_reading = input("请输入 '是' 确认完成阅读，或输入 '否' 继续阅读: ").strip()
                        if finished_reading.lower() in ['是', 'y', 'yes']:
                            print("正在为您分类并归档这篇论文...")
                            # Ask the AI to categorize the paper
                            try:
                                result = await self.session.call_tool("categorize_paper", {"filename": self.current_paper})
                                
                                # Extract result text
                                result_text = ""
                                if isinstance(result, list) and len(result) > 0:
                                    for item in result:
                                        if hasattr(item, 'text'):
                                            result_text += item.text
                                        else:
                                            result_text += str(item)
                                
                                print(f"论文已分类并归档: {result_text}")
                                
                                # Save conversation log for this paper
                                if self.conversation_history:
                                    print("正在保存对话记录...")
                                    try:
                                        # Format conversation history
                                        log_content = "论文阅读对话记录\n==================\n\n"
                                        for msg in self.conversation_history:
                                            role = msg.get("role", "unknown")
                                            content = msg.get("content", "")
                                            if role == "user":
                                                log_content += f"用户: {content}\n\n"
                                            elif role == "assistant":
                                                log_content += f"助手: {content}\n\n"
                                            elif role == "tool":
                                                log_content += f"工具结果: {content}\n\n"
                                        
                                        # Save the log
                                        log_result = await self.session.call_tool(
                                            "save_conversation_log", 
                                            {
                                                "paper_filename": self.current_paper,
                                                "log_content": log_content
                                            }
                                        )
                                        
                                        # Extract log result text
                                        log_result_text = ""
                                        if isinstance(log_result, list) and len(log_result) > 0:
                                            for item in log_result:
                                                if hasattr(item, 'text'):
                                                    log_result_text += item.text
                                                else:
                                                    log_result_text += str(item)
                                        
                                        print(f"对话记录已保存: {log_result_text}")
                                    except Exception as log_error:
                                        print(f"保存对话记录时出错: {str(log_error)}")
                                
                                # Remove from downloaded papers list if it's there
                                if self.current_paper in self.downloaded_papers:
                                    self.downloaded_papers.remove(self.current_paper)
                                self.current_paper = None
                                self.current_paper_path = None
                            except Exception as e:
                                print(f"分类论文时出错: {str(e)}")
                        else:
                            print("好的，您可以继续阅读。")
                    
                    # Check if there are other downloaded papers that might need categorization
                    if self.downloaded_papers:
                        print(f"\n本次会话中您还下载了以下论文:")
                        for i, paper in enumerate(self.downloaded_papers, 1):
                            print(f"{i}. {paper}")
                        
                        paper_indices_to_categorize = []
                        for i, paper in enumerate(self.downloaded_papers):
                            finished_reading = input(f"\n您是否已完成阅读 '{paper}'？(是/否): ").strip().lower()
                            if finished_reading in ['是', 'y', 'yes']:
                                paper_indices_to_categorize.append(i)
                        
                        # Process in reverse order to maintain indices
                        for index in reversed(paper_indices_to_categorize):
                            paper = self.downloaded_papers[index]
                            print(f"\n正在为您分类并归档论文 '{paper}'...")
                            
                            try:
                                result = await self.session.call_tool("categorize_paper", {"filename": paper})
                                
                                # Extract result text
                                result_text = ""
                                if isinstance(result, list) and len(result) > 0:
                                    for item in result:
                                        if hasattr(item, 'text'):
                                            result_text += item.text
                                        else:
                                            result_text += str(item)
                                
                                print(f"论文 '{paper}' 已分类并归档: {result_text}")
                                
                                # Save conversation log for this paper
                                if self.conversation_history:
                                    print("正在保存对话记录...")
                                    try:
                                        # Format conversation history
                                        log_content = "论文阅读对话记录\n==================\n\n"
                                        for msg in self.conversation_history:
                                            role = msg.get("role", "unknown")
                                            content = msg.get("content", "")
                                            if role == "user":
                                                log_content += f"用户: {content}\n\n"
                                            elif role == "assistant":
                                                log_content += f"助手: {content}\n\n"
                                            elif role == "tool":
                                                log_content += f"工具结果: {content}\n\n"
                                        
                                        # Save the log
                                        log_result = await self.session.call_tool(
                                            "save_conversation_log", 
                                            {
                                                "paper_filename": paper,
                                                "log_content": log_content
                                            }
                                        )
                                        
                                        # Extract log result text
                                        log_result_text = ""
                                        if isinstance(log_result, list) and len(log_result) > 0:
                                            for item in log_result:
                                                if hasattr(item, 'text'):
                                                    log_result_text += item.text
                                                else:
                                                    log_result_text += str(item)
                                        
                                        print(f"对话记录已保存: {log_result_text}")
                                    except Exception as log_error:
                                        print(f"保存对话记录时出错: {str(log_error)}")
                                
                                # Remove from downloaded papers list
                                self.downloaded_papers.pop(index)
                            except Exception as e:
                                print(f"分类论文 '{paper}' 时出错: {str(e)}")
                    
                    if not self.current_paper and not self.downloaded_papers:
                        print("\n本次会话中您没有下载任何论文。")
                    
                    print("\n助手: 再见！希望我的帮助对您有用。")
                    break
                
                print("助手: ", end="")
                response = await self.process_query_with_deepseek(user_input)
                print(response)
                    
            except KeyboardInterrupt:
                print("\n\n再见！希望我的帮助对您有用。")
                break
            except EOFError:
                print("\n\n再见！希望我的帮助对您有用。")
                break
            except Exception as e:
                print(f"\n执行过程中出错: {str(e)}")
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    assistant = PaperReadingAssistant()
    try:
        await assistant.connect_to_server()
        await assistant.interactive_session()
    finally:
        await assistant.cleanup()

if __name__ == "__main__":
    asyncio.run(main())