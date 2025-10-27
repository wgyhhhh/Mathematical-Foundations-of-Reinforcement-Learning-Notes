# 备份智能摘要代码
import re
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from functools import lru_cache

class AISummaryGenerator:
    def __init__(self):
        self.cache_dir = Path("site/.ai_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # DeepSeek API配置
        self.api_config = {
            'url': 'https://api.deepseek.com/v1/chat/completions',
            'model': 'deepseek-chat',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer sk-'
            }
        }
        
        # 📂 可自定义的文件夹配置
        self.enabled_folders = [
            'blog/',      # blog文件夹
            'develop/',   # develop文件夹
            # 'about/',    # about文件夹
            # 在这里添加您想要启用AI摘要的文件夹
        ]
        
        # 📋 排除的文件和文件夹
        self.exclude_patterns = [
            'liuyanban.md', 'link.md', '404.md', 'tag.md', 'tags.md',
            '/about/', '/search/', '/sitemap', 'index.md',  # 根目录index.md
        ]
        
        # 📋 排除的特定文件
        self.exclude_files = [
            'blog/index.md',
            'blog/indexblog.md',
            'docs/index.md',
            'develop/index.md',
        ]
    
    def configure_folders(self, folders=None, exclude_patterns=None, exclude_files=None):
        """
        配置启用AI摘要的文件夹
        
        Args:
            folders: 启用AI摘要的文件夹列表
            exclude_patterns: 排除的模式列表  
            exclude_files: 排除的特定文件列表
        """
        if folders is not None:
            self.enabled_folders = folders
        if exclude_patterns is not None:
            self.exclude_patterns = exclude_patterns
        if exclude_files is not None:
            self.exclude_files = exclude_files
    
    def get_content_hash(self, content):
        """生成内容hash用于缓存"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_cached_summary(self, content_hash):
        """获取缓存的摘要"""
        cache_file = self.cache_dir / f"{content_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 检查缓存是否过期（7天）
                    cache_time = datetime.fromisoformat(cache_data.get('timestamp', '1970-01-01'))
                    if (datetime.now() - cache_time).days < 7:
                        return cache_data
            except:
                pass
        return None
    
    def save_summary_cache(self, content_hash, summary_data):
        """保存摘要到缓存"""
        cache_file = self.cache_dir / f"{content_hash}.json"
        try:
            summary_data['timestamp'] = datetime.now().isoformat()
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存摘要缓存失败: {e}")
    
    def clean_content_for_ai(self, markdown):
        """清理内容，提取主要文本用于AI处理"""
        content = markdown
        
        # 移除YAML front matter
        content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)
        
        # 移除已存在的阅读信息块和AI摘要块
        content = re.sub(r'!!! info "📖 阅读信息".*?(?=\n\n|\n#|\Z)', '', content, flags=re.DOTALL)
        content = re.sub(r'!!! info "🤖 AI智能摘要".*?(?=\n\n|\n#|\Z)', '', content, flags=re.DOTALL)
        content = re.sub(r'!!! tip "📝 自动摘要".*?(?=\n\n|\n#|\Z)', '', content, flags=re.DOTALL)
        
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        
        # 移除图片，保留alt文本作为内容提示
        content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[图片：\1]', content)
        
        # 移除链接，保留文本
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        # 移除代码块，但保留关键信息
        content = re.sub(r'```(\w+)?\n(.*?)\n```', r'[代码示例]', content, flags=re.DOTALL)
        
        # 移除行内代码
        content = re.sub(r'`[^`]+`', '[代码]', content)
        
        # 移除表格格式但保留内容
        content = re.sub(r'\|[^\n]+\|', '', content)
        content = re.sub(r'^[-|:\s]+$', '', content, flags=re.MULTILINE)
        
        # 清理格式符号
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # 粗体
        content = re.sub(r'\*([^*]+)\*', r'\1', content)      # 斜体
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)  # 标题符号
        
        # 移除多余的空行和空格
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'^[ \t]+', '', content, flags=re.MULTILINE)
        content = content.strip()
        
        return content
    
    def generate_ai_summary(self, content, page_title=""):
        """使用DeepSeek生成摘要"""
        # 优化的提示词
        prompt = f"""请为以下技术文章生成一个高质量的摘要，要求：

1. **长度控制**：严格控制在80-120字以内
2. **内容要求**：
   - 准确概括文章的核心主题和关键要点
   - 突出技术特点、应用场景或解决的问题
   - 使用专业但易懂的语言
   - 避免重复文章标题的内容
3. **格式要求**：
   - 直接返回摘要内容，无需任何前缀或后缀
   - 使用简洁的陈述句
   - 可以适当使用技术术语

文章标题：{page_title}

文章内容：
{content[:2500]}

请生成摘要："""

        try:
            payload = {
                "model": self.api_config['model'],
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的技术文档摘要专家，擅长提取文章核心要点并生成简洁准确的摘要。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3,  # 降低随机性，提高准确性
                "top_p": 0.9
            }
            
            response = requests.post(
                self.api_config['url'],
                headers=self.api_config['headers'],
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result['choices'][0]['message']['content'].strip()
                
                # 清理可能的格式问题
                summary = re.sub(r'^["""''`]+|["""''`]+$', '', summary)
                summary = re.sub(r'^\s*摘要[：:]\s*', '', summary)
                summary = re.sub(r'^\s*总结[：:]\s*', '', summary)
                
                return summary
            else:
                print(f"DeepSeek API请求失败: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API请求异常: {e}")
            return None
        except Exception as e:
            print(f"AI摘要生成异常: {e}")
            return None
    
    def generate_fallback_summary(self, content, page_title=""):
        """生成备用摘要（基于规则的智能摘要）"""
        # 移除格式符号
        clean_text = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)
        
        # 分割成句子
        sentences = re.split(r'[\u3002\uff01\uff1f.!?]', clean_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        # 优先选择包含关键词的句子
        key_indicators = [
            '介绍', '讲解', '说明', '分析', '探讨', '研究', '实现', '应用',
            '方法', '技术', '算法', '原理', '概念', '特点', '优势', '解决',
            '教程', '指南', '配置', '安装', '部署', '开发', '设计', '构建'
        ]
        
        priority_sentences = []
        normal_sentences = []
        
        for sentence in sentences[:10]:  # 处理前10句
            if any(keyword in sentence for keyword in key_indicators):
                priority_sentences.append(sentence)
            else:
                normal_sentences.append(sentence)
        
        # 组合摘要
        selected_sentences = []
        total_length = 0
        
        # 优先使用关键句子
        for sentence in priority_sentences:
            if total_length + len(sentence) > 100:
                break
            selected_sentences.append(sentence)
            total_length += len(sentence)
        
        # 如果还有空间，添加普通句子
        if total_length < 80:
            for sentence in normal_sentences:
                if total_length + len(sentence) > 100:
                    break
                selected_sentences.append(sentence)
                total_length += len(sentence)
        
        if selected_sentences:
            summary = '.'.join(selected_sentences) + '.'
            # 简化冗长的摘要
            if len(summary) > 120:
                summary = selected_sentences[0] + '.'
            return summary
        else:
            # 根据标题生成通用摘要
            if any(keyword in page_title for keyword in ['教程', '指南', 'Tutorial']):
                return '本文提供了详细的教程指南，通过实例演示帮助读者掌握相关技术要点。'
            elif any(keyword in page_title for keyword in ['配置', '设置', '安装', 'Config']):
                return '本文介绍了系统配置的方法和步骤，提供实用的设置建议和最佳实践。'
            elif any(keyword in page_title for keyword in ['开发', '编程', 'Development']):
                return '本文分享了开发经验和技术实践，提供了实用的代码示例和解决方案。'
            else:
                return '本文深入探讨了相关技术内容，提供了实用的方法和解决方案。'
    
    def process_page(self, markdown, page, config):
        """处理页面，生成AI摘要"""
        if not self.should_generate_summary(page, markdown):
            return markdown
        
        clean_content = self.clean_content_for_ai(markdown)
        
        # 内容长度检查
        if len(clean_content) < 200:
            print(f"📄 内容太短，跳过摘要生成: {page.file.src_path}")
            return markdown
        
        content_hash = self.get_content_hash(clean_content)
        page_title = getattr(page, 'title', '')
        
        # 检查缓存
        cached_summary = self.get_cached_summary(content_hash)
        if cached_summary:
            summary = cached_summary.get('summary', '')
            ai_service = 'cached'
            print(f"✅ 使用缓存摘要: {page.file.src_path}")
        else:
            # 生成新摘要
            print(f"🤖 正在生成AI摘要: {page.file.src_path}")
            summary = self.generate_ai_summary(clean_content, page_title)
            
            if not summary:
                summary = self.generate_fallback_summary(clean_content, page_title)
                ai_service = 'fallback'
                print(f"📝 使用备用摘要: {page.file.src_path}")
            else:
                ai_service = 'deepseek'
                print(f"✅ AI摘要生成成功: {page.file.src_path}")
            
            # 保存到缓存
            self.save_summary_cache(content_hash, {
                'summary': summary,
                'service': ai_service,
                'page_title': page_title
            })
        
        # 添加摘要到页面最上面
        summary_html = self.format_summary(summary, ai_service)
        return summary_html + '\n\n' + markdown
    
    def should_generate_summary(self, page, markdown):
        """判断是否应该生成摘要 - 可自定义文件夹"""
        # 检查页面元数据
        if hasattr(page, 'meta'):
            # 明确禁用
            if page.meta.get('ai_summary') == False:
                return False
            
            # 强制启用
            if page.meta.get('ai_summary') == True:
                return True
        
        # 获取文件路径
        src_path = page.file.src_path.replace('\\', '/')  # 统一路径分隔符
        
        # 检查排除模式
        if any(pattern in src_path for pattern in self.exclude_patterns):
            return False
        
        # 检查排除的特定文件
        if src_path in self.exclude_files:
            return False
        
        # 检查是否在启用的文件夹中
        for folder in self.enabled_folders:
            if src_path.startswith(folder) or f'/{folder}' in src_path:
                folder_name = folder.rstrip('/')
                print(f"🎯 {folder_name}文件夹文章检测到，启用AI摘要: {src_path}")
                return True
        
        # 默认不生成摘要
        return False
    
    def format_summary(self, summary, ai_service):
        """格式化摘要显示"""
        service_config = {
            'deepseek': {
                'icon': '🤖',
                'name': 'AI智能摘要',
                'color': 'info'
            },
            'fallback': {
                'icon': '📝',
                'name': '自动摘要',
                'color': 'tip'
            },
            'cached': {
                'icon': '💾',
                'name': 'AI智能摘要',
                'color': 'info'
            }
        }
        
        config = service_config.get(ai_service, service_config['deepseek'])
        
        return f'''??? {config['color']} "{config['icon']} {config['name']}"
    {summary}

'''

# 创建全局实例
ai_summary_generator = AISummaryGenerator()

# 🔧 自定义配置函数
def configure_ai_summary(enabled_folders=None, exclude_patterns=None, exclude_files=None):
    """
    配置AI摘要功能
    
    Args:
        enabled_folders: 启用AI摘要的文件夹列表，例如 ['blog/', 'docs/', 'posts/']
        exclude_patterns: 排除的模式列表，例如 ['404.md', '/admin/']
        exclude_files: 排除的特定文件列表，例如 ['blog/index.md']
    
    Example:
        # 只在blog和docs文件夹启用
        configure_ai_summary(['blog/', 'docs/'])
        
        # 在所有文件夹启用，但排除特定文件
        configure_ai_summary([''], exclude_files=['index.md', 'about.md'])
    """
    ai_summary_generator.configure_folders(enabled_folders, exclude_patterns, exclude_files)

def on_page_markdown(markdown, page, config, files):
    """MkDocs hook入口点"""
    return ai_summary_generator.process_page(markdown, page, config)

