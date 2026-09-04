import logging
import mcp.types as types
from typing import Dict, Any, List
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
from mcp.server.stdio import stdio_server
import os
import json
import arxiv
import PyPDF2
import pdfplumber
import asyncio
from pathlib import Path
import googlesearch
from dotenv import load_dotenv
from openai import OpenAI
import urllib.request
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("paper-reader-mcp-server")
logger.setLevel(logging.INFO)

# Initialize DeepSeek client
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
) if DEEPSEEK_API_KEY else None

# Create papers directory structure
base_path = Path("./papers")
categories = ["机器学习", "强化学习", "图像生成", "大语言模型", "多模态", "其它"]

# Create base directories
for dir_path in [base_path, base_path / "unread", base_path / "read"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Create category subdirectories under both read and unread
for category in categories:
    for parent in ["unread", "read"]:
        (base_path / parent / category).mkdir(parents=True, exist_ok=True)

# Initialize server
server = Server("paper-reader-assistant")

# Define tools
search_papers_tool = types.Tool(
    name="search_papers",
    description="根据关键词搜索arXiv上的论文",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "返回结果数量，默认为5", "default": 5}
        },
        "required": ["query"]
    }
)

download_paper_tool = types.Tool(
    name="download_paper",
    description="根据arXiv ID下载论文PDF",
    inputSchema={
        "type": "object",
        "properties": {
            "paper_id": {"type": "string", "description": "arXiv论文ID"}
        },
        "required": ["paper_id"]
    }
)

extract_paper_content_tool = types.Tool(
    name="extract_paper_content",
    description="提取PDF文件中的文本内容",
    inputSchema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "PDF文件路径"}
        },
        "required": ["file_path"]
    }
)

categorize_paper_tool = types.Tool(
    name="categorize_paper",
    description="对论文进行分类并移动到已读目录",
    inputSchema={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "论文文件名"}
        },
        "required": ["filename"]
    }
)

auto_categorize_paper_tool = types.Tool(
    name="auto_categorize_paper",
    description="自动对下载的论文进行分类",
    inputSchema={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "论文文件名"}
        },
        "required": ["filename"]
    }
)

search_paper_explanation_tool = types.Tool(
    name="search_paper_explanation",
    description="搜索论文的相关解析文章",
    inputSchema={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "论文标题"}
        },
        "required": ["title"]
    }
)

organize_papers_tool = types.Tool(
    name="organize_papers",
    description="整理本地论文，处理重复和未分类的论文",
    inputSchema={
        "type": "object",
        "properties": {},
        "required": []
    }
)

mark_as_read_tool = types.Tool(
    name="mark_as_read",
    description="将论文标记为已读（移动到read目录）",
    inputSchema={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "论文文件名"}
        },
        "required": ["filename"]
    }
)

research_assistant_tool = types.Tool(
    name="research_assistant",
    description="研究助手：根据主题进行全面研究",
    inputSchema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "研究主题"}
        },
        "required": ["topic"]
    }
)

save_conversation_log_tool = types.Tool(
    name="save_conversation_log",
    description="保存当前对话记录到指定论文的目录中",
    inputSchema={
        "type": "object",
        "properties": {
            "paper_filename": {"type": "string", "description": "论文文件名"},
            "log_content": {"type": "string", "description": "对话记录内容"}
        },
        "required": ["paper_filename", "log_content"]
    }
)

@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """List available tools."""
    return [
        search_papers_tool,
        download_paper_tool,
        extract_paper_content_tool,
        categorize_paper_tool,
        auto_categorize_paper_tool,
        search_paper_explanation_tool,
        organize_papers_tool,
        mark_as_read_tool,
        research_assistant_tool,
        save_conversation_log_tool
    ]

async def search_papers(query: str, max_results: int = 5) -> str:
    """Search papers on arXiv."""
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "id": result.get_short_id(),
                "pdf_url": result.pdf_url
            }
            papers.append(paper)
            
        result_str = json.dumps(papers, ensure_ascii=False)
        return f"已在arXiv上搜索'{query}'，找到{len(papers)}篇相关论文。搜索结果：\n{result_str}"
    except Exception as e:
        logger.error(f"搜索论文时出错: {str(e)}")
        return json.dumps([])

async def download_paper(paper_id: str) -> str:
    """Download paper by arXiv ID."""
    try:
        file_path = base_path / "unread" / f"{paper_id}.pdf"
        
        if file_path.exists():
            logger.info(f"论文 {paper_id} 已存在，无需重新下载")
            # Still attempt to categorize if needed
            try:
                filename = f"{paper_id}.pdf"
                categorize_result = await auto_categorize_paper(filename)
                return f"论文 {paper_id} 已存在，路径为: {str(file_path)}\n{categorize_result}"
            except Exception as categorize_error:
                logger.error(f"自动分类已有论文时出错: {str(categorize_error)}")
                return f"论文 {paper_id} 已存在，路径为: {str(file_path)}"
            
        search = arxiv.Search(id_list=[paper_id])
        results = list(search.results())
        
        if not results:
            raise Exception(f"未找到ID为 {paper_id} 的论文")
            
        paper = results[0]
        
        # Try default download method first
        try:
            paper.download_pdf(dirpath=str(base_path / "unread"), filename=f"{paper_id}.pdf")
            logger.info(f"成功下载论文 {paper_id}")
        except Exception as e:
            logger.warning(f"使用默认方法下载失败: {e}, 尝试手动下载")
            # Manual download with custom SSL context
            pdf_url = paper.pdf_url
            if not pdf_url:
                raise Exception(f"论文 {paper_id} 没有可用的PDF链接")
                
            # Create SSL context that doesn't verify certificates (for problematic sites)
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            try:
                # Use requests library if available, otherwise fall back to urllib
                try:
                    import requests
                    response = requests.get(pdf_url, headers=headers, timeout=30, verify=False)
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                except ImportError:
                    # Fallback to urllib with custom SSL context
                    import urllib.request
                    request = urllib.request.Request(pdf_url, headers=headers)
                    with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
                        with open(file_path, 'wb') as f:
                            f.write(response.read())
                logger.info(f"手动下载论文 {paper_id} 成功")
            except Exception as manual_e:
                logger.error(f"手动下载也失败了: {manual_e}")
                raise Exception(f"下载论文失败: {manual_e}")
        
        if not file_path.exists():
            raise Exception(f"下载似乎成功，但文件未找到: {file_path}")
            
        # Automatically categorize the paper after downloading
        try:
            filename = f"{paper_id}.pdf"
            categorize_result = await auto_categorize_paper(filename)
            return f"已成功下载论文至: {str(file_path)}\n{categorize_result}"
        except Exception as categorize_error:
            logger.error(f"自动分类论文时出错: {str(categorize_error)}")
            return f"已成功下载论文至: {str(file_path)}\n注意：自动分类失败: {str(categorize_error)}"
    except Exception as e:
        logger.error(f"下载论文时出错: {str(e)}")
        raise Exception(f"下载论文失败: {str(e)}")

async def extract_paper_content(file_path: str) -> str:
    """Extract text content from PDF file."""
    try:
        if not Path(file_path).exists():
            possible_paths = [
                base_path / "unread" / Path(file_path).name,
                base_path / "read" / Path(file_path).name
            ]
            
            # Check in category directories
            for parent in ["unread", "read"]:
                for category in categories:
                    possible_paths.append(base_path / parent / category / Path(file_path).name)
            
            for path in possible_paths:
                if path.exists():
                    file_path = str(path)
                    break
            else:
                raise Exception(f"找不到文件: {file_path}")
        
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text  # Always return raw text
    except Exception as e:
        logger.error(f"提取PDF内容时出错: {str(e)}")
        try:
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text  # Always return raw text
        except Exception as e2:
            logger.error(f"使用PyPDF2也失败了: {str(e2)}")
            raise Exception(f"提取PDF内容失败: {str(e)}")

async def auto_categorize_paper(filename: str) -> str:
    """Automatically categorize paper after downloading."""
    try:
        # Find the paper in unread directory
        source_path = base_path / "unread" / filename
        if not source_path.exists():
            # Check categorized directories
            found = False
            for category in categories:
                cat_path = base_path / "unread" / category / filename
                if cat_path.exists():
                    source_path = cat_path
                    found = True
                    break
            if not found:
                raise Exception(f"找不到待分类的论文: {filename}")
        
        # If already in a categorized directory, just return
        if source_path.parent.name in categories:
            return f"论文 {filename} 已在分类目录: {source_path.parent.name}"
        
        # Extract content for categorization (first ~500 characters which usually includes title and abstract)
        content = await extract_paper_content(str(source_path))
        abstract = content[:500]  # Just need a small portion for categorization
        
        # Ask AI to categorize the paper
        if deepseek_client:
            try:
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一位专业的学术分类专家，请根据论文标题和摘要判断该论文属于以下哪个领域。"},
                        {"role": "user", "content": f"请判断这篇论文属于以下哪个领域：机器学习、强化学习、图像生成、大语言模型、多模态、其它。只回答领域名称。\n\n论文标题和摘要：{abstract}"}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                
                category = response.choices[0].message.content.strip()
                if category not in categories:
                    category = "其它"  # Default to "其它" if not matched
            except Exception as e:
                logger.error(f"DeepSeek API调用失败: {str(e)}")
                category = "其它"
        else:
            category = "其它"
        
        # Move to categorized unread directory
        target_dir = base_path / "unread" / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        
        # Only move if source and target are different
        if source_path != target_file:
            source_path.rename(target_file)
            return f"已将论文 {filename} 自动分类到: {category}"
        else:
            return f"论文 {filename} 已在正确分类目录: {category}"
    except Exception as e:
        logger.error(f"自动分类论文时出错: {str(e)}")
        raise Exception(f"自动分类论文失败: {str(e)}")

async def categorize_paper(filename: str) -> str:
    """Move paper from unread to read directory with appropriate category."""
    try:
        # First check if the file is already in read directory
        for category in ["机器学习", "强化学习", "图像生成", "大语言模型", "多模态", "其它"]:
            category_file = base_path / "read" / category / filename
            if category_file.exists():
                return f"论文 {filename} 已经在已读目录的 {category} 分类中"
        
        # Look for the file in unread directories
        source_path = None
        source_category = None
        
        # Check in uncategorized unread
        unread_file = base_path / "unread" / filename
        if unread_file.exists():
            source_path = unread_file
        else:
            # Check in categorized unread
            for category in ["机器学习", "强化学习", "图像生成", "大语言模型", "多模态", "其它"]:
                category_file = base_path / "unread" / category / filename
                if category_file.exists():
                    source_path = category_file
                    source_category = category
                    break
        
        if not source_path:
            raise Exception(f"找不到文件: {filename}")
        
        # If we don't know the category yet, determine it now
        if not source_category:
            # Extract content for categorization
            content = await extract_paper_content(str(source_path))
            abstract = content[:500]  # Just need a small portion for categorization
            
            # Ask AI to categorize the paper
            if deepseek_client:
                try:
                    response = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一位专业的学术分类专家，请根据论文标题和摘要判断该论文属于以下哪个领域。"},
                            {"role": "user", "content": f"请判断这篇论文属于以下哪个领域：机器学习、强化学习、图像生成、大语言模型、多模态、其它。只回答领域名称。\n\n论文标题和摘要：{abstract}"}
                        ],
                        max_tokens=100,
                        temperature=0.1
                    )
                    
                    source_category = response.choices[0].message.content.strip()
                    categories = ["机器学习", "强化学习", "图像生成", "大语言模型", "多模态", "其它"]
                    if source_category not in categories:
                        source_category = "其它"
                except Exception as e:
                    logger.error(f"DeepSeek API调用失败: {str(e)}")
                    source_category = "其它"
            else:
                source_category = "其它"
        
        # Move to read directory with category
        target_dir = base_path / "read" / source_category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        
        source_path.rename(target_file)
        
        return f"已将论文 {filename} 从未读目录移动到已读目录，并分类到 {source_category} 类别中"
    except Exception as e:
        logger.error(f"分类论文时出错: {str(e)}")
        return f"分类论文失败: {str(e)}"

async def search_paper_explanation(title: str) -> str:
    """Search for paper explanations using Baidu."""
    try:
        import requests
        import urllib.parse
        import time
        import random
        
        # Try multiple search queries to improve results
        search_queries = [
            f"{title} 解析",
            f"{title} 教程",
            f"{title} 分析",
            f"{title} 解读",
            f'"{title}" 技术博客',
        ]
        
        all_results = []
        
        # Headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.baidu.com/',
        }
        
        # Try each search query until we find results
        for query in search_queries:
            try:
                encoded_query = urllib.parse.quote(query)
                search_url = f"https://www.baidu.com/s?wd={encoded_query}"
                
                # Send request with retry mechanism and longer timeout
                response = None
                last_exception = None
                for attempt in range(2):  # Try twice for each query
                    try:
                        response = requests.get(search_url, headers=headers, timeout=15)
                        response.raise_for_status()
                        break
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"搜索 '{query}' 第{attempt+1}次请求失败: {str(e)}")
                        last_exception = e
                        if attempt < 1:
                            time.sleep(1 + random.uniform(0, 1))
                        else:
                            break
                
                if not response:
                    continue  # Try next query
                
                # Log response info for debugging
                logger.info(f"搜索 '{query}' 成功，响应长度: {len(response.text)}")
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract search results using multiple strategies
                results = []
                
                # Strategy 1: Look for result containers with data-tools attribute
                result_containers = soup.find_all('div', attrs={'data-tools': True})
                for container in result_containers[:8]:
                    try:
                        # Extract data from data-tools attribute
                        data_tools = container.get('data-tools', '')
                        if data_tools:
                            import re
                            # Try to extract title and URL using regex
                            title_match = re.search(r'"title":"([^"]*)"', data_tools)
                            url_match = re.search(r'"url":"([^"]*)"', data_tools)
                            
                            if title_match and url_match:
                                title_text = title_match.group(1).encode('utf-8').decode('unicode_escape')
                                url = url_match.group(1).encode('utf-8').decode('unicode_escape')
                                
                                # Clean up the title
                                title_text = ' '.join(title_text.split())
                                if 8 < len(title_text) < 150 and 'baidu.com' not in url:
                                    results.append({
                                        'title': title_text,
                                        'url': urllib.parse.unquote(url),
                                        'query': query  # Track which query produced this result
                                    })
                    except Exception as parse_error:
                        logger.warning(f"解析data-tools属性时出错: {str(parse_error)}")
                        continue
                
                # Strategy 2: Look for traditional h3/a structure
                if len(results) < 3:
                    h3_tags = soup.find_all('h3', class_='t')
                    for h3 in h3_tags[:8]:
                        try:
                            link_tag = h3.find('a')
                            if link_tag and link_tag.get('href'):
                                href = link_tag.get('href')
                                title_text = link_tag.get_text(strip=True)
                                
                                # Validate link and title
                                if href and title_text:
                                    # Resolve redirect links if needed
                                    if href.startswith('http'):
                                        resolved_href = href
                                    else:
                                        # Try to resolve relative links
                                        try:
                                            resolved_href = urllib.parse.urljoin('https://www.baidu.com', href)
                                        except:
                                            resolved_href = href
                                    
                                    # Clean up the title
                                    title_text = ' '.join(title_text.split())
                                    if 8 < len(title_text) < 150 and 'baidu.com' not in resolved_href:
                                        results.append({
                                            'title': title_text,
                                            'url': resolved_href,
                                            'query': query  # Track which query produced this result
                                        })
                        except Exception as parse_error:
                            logger.warning(f"解析传统结构时出错: {str(parse_error)}")
                            continue
                
                # Strategy 3: General fallback approach
                if len(results) < 2:
                    links = soup.find_all('a', href=True)
                    for link in links[:20]:  # Check more links
                        try:
                            href = link['href']
                            text = link.get_text(strip=True)
                            if href and text:
                                # Check if it looks like a search result link
                                if 'baidu.com/link' in href or ('http' in href and 'baidu.com' not in href):
                                    # Resolve redirect links if needed
                                    if href.startswith('http'):
                                        resolved_href = href
                                    else:
                                        try:
                                            resolved_href = urllib.parse.urljoin('https://www.baidu.com', href)
                                        except:
                                            resolved_href = href
                                    
                                    text = ' '.join(text.split())
                                    # Looser constraints for title length
                                    if 8 < len(text) < 150 and 'baidu.com' not in resolved_href:
                                        # Avoid obvious ads or navigation links
                                        if not any(ad_word in text.lower() for ad_word in ['广告', '推广', '更多', '首页', '导航', '登录', '注册']):
                                            results.append({
                                                'title': text,
                                                'url': resolved_href,
                                                'query': query  # Track which query produced this result
                                            })
                        except Exception as parse_error:
                            logger.warning(f"解析通用结构时出错: {str(parse_error)}")
                            continue
                
                # Add results to all_results if we found any
                if results:
                    all_results.extend(results)
                    # If we have enough results, break
                    if len(all_results) >= 3:
                        break
                        
            except Exception as query_error:
                logger.warning(f"处理搜索查询 '{query}' 时出错: {str(query_error)}")
                continue
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            # Normalize URL for comparison
            try:
                parsed_url = urllib.parse.urlparse(result['url'])
                normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            except:
                normalized_url = result['url'].split('?')[0]  # Fallback to simple method
            
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)
        
        # Take only top results (max 5)
        unique_results = unique_results[:5]
        
        # Format results
        if unique_results:
            formatted_results = "为您找到以下相关解析文章：\n"
            for i, result in enumerate(unique_results, 1):
                formatted_results += f"{i}. {result['title']}\n   链接: {result['url']}\n"
                # Show which query produced this result (for debugging)
                formatted_results += f"   搜索词: {result['query']}\n\n"
            
            # Ask AI to determine which ones are most relevant
            if deepseek_client:
                try:
                    ai_prompt = f"以下是关于'{title}'的搜索结果，请分析并推荐1-2个最权威和相关的解析文章：\n\n"
                    for i, result in enumerate(unique_results, 1):
                        ai_prompt += f"{i}. {result['title']}\n   链接: {result['url']}\n\n"
                    ai_prompt += "请只回复你认为最相关的1-2个文章编号，例如'1, 3'。"
                    
                    ai_response = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一位专业的学术研究助理，善于判断技术文章的权威性和相关性。"},
                            {"role": "user", "content": ai_prompt}
                        ],
                        max_tokens=50,
                        temperature=0.1
                    )
                    
                    recommended_indices = ai_response.choices[0].message.content.strip()
                    formatted_results += f"\nAI推荐的权威解析文章: {recommended_indices}\n"
                except Exception as ai_error:
                    logger.warning(f"AI分析推荐失败: {str(ai_error)}")
            
            return f"{formatted_results}\n您可以点击以上链接查看详细内容。"
        else:
            # Fallback to general search advice
            fallback_queries = [f"{title} 解析", f"{title} 技术博客", f"{title} tutorial"]
            fallback_info = "\n".join([f"  - https://www.baidu.com/s?wd={urllib.parse.quote(q)}" for q in fallback_queries])
            return f"未能找到具体的解析文章。您可以尝试以下搜索：\n{fallback_info}"
            
    except Exception as e:
        logger.error(f"搜索论文解析时出错: {str(e)}")
        fallback_url = f"https://www.baidu.com/s?wd={urllib.parse.quote(title + ' 解析')}"
        return f"搜索解析文章时遇到问题: {str(e)}\n您可以手动在以下链接搜索：{title} 相关解析\n搜索链接: {fallback_url}"

async def organize_papers() -> str:
    """Organize local papers: remove unclassified papers in read folder and duplicates."""
    try:
        removed_files = []
        
        # 1. Remove unclassified papers in read folder
        read_root_path = base_path / "read"
        if read_root_path.exists():
            for file_path in read_root_path.iterdir():
                if file_path.is_file() and file_path.suffix == ".pdf":
                    file_path.unlink()
                    removed_files.append(f"已删除read目录中未分类的论文: {file_path.name}")
        
        # 2. Find and remove duplicates between read and unread folders
        # First, collect all papers in read directory (including subdirectories)
        read_papers = set()
        for category in categories:
            category_path = read_root_path / category
            if category_path.exists():
                for file_path in category_path.iterdir():
                    if file_path.is_file() and file_path.suffix == ".pdf":
                        read_papers.add(file_path.name)
        
        # Then, check unread directory and remove duplicates
        unread_root_path = base_path / "unread"
        if unread_root_path.exists():
            # Check uncategorized unread papers
            for file_path in unread_root_path.iterdir():
                if file_path.is_file() and file_path.suffix == ".pdf":
                    if file_path.name in read_papers:
                        file_path.unlink()
                        removed_files.append(f"已删除unread目录中的重复论文: {file_path.name}")
            
            # Check categorized unread papers
            for category in categories:
                category_path = unread_root_path / category
                if category_path.exists():
                    for file_path in category_path.iterdir():
                        if file_path.is_file() and file_path.suffix == ".pdf":
                            if file_path.name in read_papers:
                                file_path.unlink()
                                removed_files.append(f"已删除unread/{category}目录中的重复论文: {file_path.name}")
        
        if removed_files:
            result = "论文整理完成:\n" + "\n".join(removed_files)
        else:
            result = "论文整理完成，未发现需要处理的文件。"
            
        return result
    except Exception as e:
        logger.error(f"整理论文时出错: {str(e)}")
        return f"整理论文失败: {str(e)}"

async def mark_as_read(filename: str) -> str:
    """Mark paper as read and move to appropriate category."""
    try:
        # First check if the file is already in read directory
        for category in ["机器学习", "强化学习", "图像生成", "大语言模型", "多模态", "其它"]:
            category_file = base_path / "read" / category / filename
            if category_file.exists():
                return f"论文 {filename} 已经在已读目录的 {category} 分类中"
        
        # Look for the file in unread directories
        source_path = None
        
        # Check in uncategorized unread
        unread_file = base_path / "unread" / filename
        if unread_file.exists():
            source_path = unread_file
        else:
            # Check in categorized unread
            for category in ["机器学习", "强化学习", "图像生成", "大语言模型", "多模态", "其它"]:
                category_file = base_path / "unread" / category / filename
                if category_file.exists():
                    source_path = category_file
                    break
        
        if not source_path:
            # Check if file is already in read directory but not categorized
            read_file = base_path / "read" / filename
            if read_file.exists():
                source_path = read_file
            else:
                raise Exception(f"找不到文件: {filename}")
        
        # Extract content for categorization
        content = await extract_paper_content(str(source_path))
        abstract = content[:1500]
        
        # Ask AI to categorize the paper
        if deepseek_client:
            try:
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一位专业的学术分类专家，请根据论文标题和摘要判断该论文属于以下哪个领域。"},
                        {"role": "user", "content": f"请判断这篇论文属于以下哪个领域：机器学习、强化学习、图像生成、大语言模型、多模态、其它。只回答领域名称。\n\n论文标题和摘要：{abstract}"}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                
                category = response.choices[0].message.content.strip()
                categories = ["机器学习", "强化学习", "图像生成", "大语言模型", "多模态", "其它"]
                if category not in categories:
                    category = "其它"  # Default to "其它" if not matched
            except Exception as e:
                logger.error(f"DeepSeek API调用失败: {str(e)}")
                category = "其它"
        else:
            category = "其它"
        
        # Move to read directory with category
        target_dir = base_path / "read" / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        
        # Only move if source and target are different
        if source_path != target_file:
            source_path.rename(target_file)
        
        return f"已将论文 {filename} 标记为已读，并分类到 {category} 目录中"
    except Exception as e:
        logger.error(f"标记为已读时出错: {str(e)}")
        return f"标记为已读失败: {str(e)}"

async def research_assistant(topic: str) -> str:
    """Research assistant for comprehensive topic analysis."""
    try:
        search_result = await search_papers(topic, 5)
        papers = json.loads(search_result) if isinstance(search_result, str) else search_result
        
        if not papers:
            return json.dumps({"error": "未找到相关论文"}, ensure_ascii=False)
        
        core_paper_analyses = []
        for i, paper in enumerate(papers[:3]):
            paper_id = paper['id']
            
            file_path = await download_paper(paper_id)
            content = await extract_paper_content(file_path)
            
            # Extract the filename for categorization
            filename = f"{paper_id}.pdf"
            
            # Use the new categorization method
            category_result = await auto_categorize_paper(filename)
            
            # Then move to read directory
            final_category_result = await categorize_paper(filename)
            
            explanations_result = await search_paper_explanation(paper['title'])
            
            summary = ""
            if deepseek_client and content:
                try:
                    content_summary = content[:2000]
                    response = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一位专业的学术研究员。请根据提供的论文内容，用中文生成一篇简洁的论文总结。"},
                            {"role": "user", "content": f"请为下面的论文生成一份中文总结：\n\n{content_summary}"}
                        ],
                        max_tokens=800,
                        temperature=0.3
                    )
                    summary = response.choices[0].message.content
                except Exception as e:
                    logger.warning(f"生成AI总结时出错: {str(e)}")
                    summary = "无法生成AI总结"
            
            core_paper_analyses.append({
                "paper_info": paper,
                "category": final_category_result,
                "ai_summary": summary,
                "explanations": explanations_result
            })
        
        final_report = {
            "research_topic": topic,
            "papers_found": len(papers),
            "core_papers": core_paper_analyses
        }
        
        return json.dumps(final_report, ensure_ascii=False)
    
    except Exception as e:
        logger.error(f"研究助手执行出错: {str(e)}")
        return json.dumps({"error": f"研究助手执行出错: {str(e)}"}, ensure_ascii=False)

async def save_conversation_log(paper_filename: str, log_content: str) -> str:
    """Save conversation log to the paper's directory."""
    try:
        # Find the paper in all possible directories
        paper_path = None
        paper_category = None
        
        # Check in read directories first (including categorized)
        for category in categories:
            category_path = base_path / "read" / category / paper_filename
            if category_path.exists():
                paper_path = category_path
                paper_category = category
                break
        
        # If not found in read, check in unread directories
        if not paper_path:
            # Check uncategorized unread
            uncategorized_path = base_path / "unread" / paper_filename
            if uncategorized_path.exists():
                paper_path = uncategorized_path
            else:
                # Check categorized unread
                for category in categories:
                    category_path = base_path / "unread" / category / paper_filename
                    if category_path.exists():
                        paper_path = category_path
                        paper_category = category
                        break
        
        if not paper_path or not paper_path.exists():
            # Check in translated directory as a last resort
            translated_path = base_path / "translated" / paper_filename
            if translated_path.exists():
                paper_path = translated_path
            else:
                raise Exception(f"找不到论文: {paper_filename}")
        
        # Save conversation log in the same directory as the paper
        log_path = paper_path.parent / f"{paper_filename}_conversation.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        return f"已将对话记录保存至: {log_path}"
    except Exception as e:
        logger.error(f"保存对话记录时出错: {str(e)}")
        raise Exception(f"保存对话记录失败: {str(e)}")

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls."""
    logger.debug(f"Calling tool {name} with arguments {arguments}")
    try:
        if name == "search_papers":
            result = await search_papers(arguments["query"], arguments.get("max_results", 5))
            return [types.TextContent(type="text", text=result)]
        elif name == "download_paper":
            result = await download_paper(arguments["paper_id"])
            return [types.TextContent(type="text", text=result)]
        elif name == "extract_paper_content":
            result = await extract_paper_content(arguments["file_path"])
            return [types.TextContent(type="text", text=result)]
        elif name == "categorize_paper":
            result = await categorize_paper(arguments["filename"])
            return [types.TextContent(type="text", text=result)]
        elif name == "auto_categorize_paper":
            result = await auto_categorize_paper(arguments["filename"])
            return [types.TextContent(type="text", text=result)]
        elif name == "search_paper_explanation":
            result = await search_paper_explanation(arguments["title"])
            return [types.TextContent(type="text", text=result)]
        elif name == "organize_papers":
            result = await organize_papers()
            return [types.TextContent(type="text", text=result)]
        elif name == "mark_as_read":
            result = await mark_as_read(arguments["filename"])
            return [types.TextContent(type="text", text=result)]
        elif name == "research_assistant":
            result = await research_assistant(arguments["topic"])
            return [types.TextContent(type="text", text=result)]
        elif name == "save_conversation_log":
            result = await save_conversation_log(arguments["paper_filename"], arguments["log_content"])
            return [types.TextContent(type="text", text=result)]
        else:
            return [types.TextContent(type="text", text=f"Error: Unknown tool {name}")]
    except Exception as e:
        logger.error(f"Tool error: {str(e)}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Run the server."""
    async with stdio_server() as streams:
        await server.run(
            streams[0],
            streams[1],
            InitializationOptions(
                server_name="paper-reader-assistant",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(resources_changed=True),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())