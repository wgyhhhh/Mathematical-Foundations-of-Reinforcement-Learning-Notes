# ---
# title: 页面标题
# disable_comments: true  # 禁用评论
# ---

from textwrap import dedent
import re

# 配置：需要添加评论的目录
COMMENT_DIRECTORIES = ['blog/', 'develop/']

# 配置：排除评论的页面列表
EXCLUDED_PAGES = {
    'blog/index.md',
    'develop/index.md',
}

# 配置：排除评论的页面模式
EXCLUDED_PATTERNS = [
    r'.*\/index\.md$',  # 排除所有 index.md 文件
    r'.*\/archive\.md$',  # 排除所有 archive.md 文件
    r'blog/category/.*\.md$',  # 排除所有 blog/category/ 下的 Markdown 文件
]

def is_page_excluded(file_path):
    """检查页面是否应该排除评论"""
    # 精确匹配检查
    if file_path in EXCLUDED_PAGES:
        return True
    
    # 模式匹配检查
    for pattern in EXCLUDED_PATTERNS:
        if re.match(pattern, file_path):
            return True
    
    return False

def should_add_comments(file_path):
    """检查文件是否应该添加评论"""
    if not file_path.endswith('.md'):
        return False
    
    # 先检查是否被排除
    if is_page_excluded(file_path):
        return False
    
    # 检查是否在指定目录下
    for directory in COMMENT_DIRECTORIES:
        if file_path.startswith(directory):
            return True
    
    return False

def on_page_markdown(markdown, **kwargs):
    """为符合条件的页面添加 Twikoo 评论系统"""
    page = kwargs['page']
    config = kwargs['config']
    
    # 检查是否应该添加评论
    if not should_add_comments(page.file.src_path):
        return markdown
    
    # 检查页面元数据是否禁用评论
    try:
        # 如果页面的 front matter 中设置了 disable_comments: true，则跳过
        if hasattr(page, 'meta') and page.meta.get('disable_comments', False):
            return markdown
    except Exception:
        pass
    
    # Twikoo 评论系统的 HTML 和 JavaScript
    twikoo_html = dedent("""
    
    <!-- Twikoo 评论系统 -->
    <div class="twikoo-container" style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(0,0,0,0.1);">
        <h3 style="margin-bottom: 1rem; font-size: 1.2rem; font-weight: 500;">💬 评论</h3>
        <div id="tcomment" class="loading" style="min-height: 200px;">
            <p style="text-align: center; color: #666; padding: 2rem;">评论系统加载中...</p>
        </div>
    </div>
    
    <style>
    .twikoo-container {
        max-width: 100%;
    }
    
    /* 暗色模式适配 */
    [data-md-color-scheme="slate"] .twikoo-container {
        border-top-color: rgba(255,255,255,0.1);
    }
    
    /* 加载状态样式 */
    #tcomment.loading {
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,0,0,0.02);
        border-radius: 8px;
    }
    
    [data-md-color-scheme="slate"] #tcomment.loading {
        background: rgba(255,255,255,0.05);
    }
    </style>
    
    <script>
    (function() {
        // 防止重复加载
        if (window.twikooLoaded) {
            return;
        }
        
        function loadTwikoo() {
            // 检查是否已经加载过 Twikoo
            if (window.twikoo) {
                initTwikoo();
                return;
            }
            
            const script = document.createElement('script');
            script.src = 'https://registry.npmmirror.com/twikoo/1.6.44/files/dist/twikoo.min.js';
            script.onload = function() {
                console.log('Twikoo 脚本加载成功');
                initTwikoo();
            };
            script.onerror = function() {
                console.error('Twikoo 脚本加载失败');
                const commentEl = document.getElementById('tcomment');
                if (commentEl) {
                    commentEl.classList.remove('loading');
                    commentEl.innerHTML = '<p style="text-align: center; color: #f56565; padding: 2rem;">评论系统加载失败，请刷新页面重试</p>';
                }
            };
            document.head.appendChild(script);
        }
        
        function initTwikoo() {
            const commentEl = document.getElementById('tcomment');
            if (!commentEl) {
                console.warn('评论容器未找到');
                return;
            }
            
            commentEl.classList.remove('loading');
            
            try {
                twikoo.init({
                    envId: 'https://superb-salamander-e730b6.netlify.app/.netlify/functions/twikoo',
                    el: '#tcomment',
                    lang: 'zh-CN',
                    path: location.pathname,
                    onCommentLoaded: function () {
                        console.log('评论加载完成');
                    },
                    onError: function(err) {
                        console.error('Twikoo 初始化失败:', err);
                        commentEl.innerHTML = '<p style="text-align: center; color: #f56565; padding: 2rem;">评论系统初始化失败，请检查网络连接</p>';
                    }
                });
                window.twikooLoaded = true;
            } catch (error) {
                console.error('Twikoo 初始化异常:', error);
                commentEl.innerHTML = '<p style="text-align: center; color: #f56565; padding: 2rem;">评论系统初始化异常</p>';
            }
        }
        
        // 页面加载完成后初始化
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', loadTwikoo);
        } else {
            loadTwikoo();
        }
    })();
    </script>
    """)
    
    return markdown.rstrip() + twikoo_html

def on_page_content(html, **kwargs):
    """在页面内容处理完成后添加必要的头部资源"""
    page = kwargs['page']
    
    # 检查是否应该添加评论
    if not should_add_comments(page.file.src_path):
        return html
    
    # 添加 KaTeX 支持（如果需要的话）
    katex_resources = """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css" integrity="sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js" integrity="sha384-g7c+Jr9ZivxKLnZTDUhnkOnsh30B4H0rpLUpJ4jAIKs4fnJI+sEnkvrMWph2EDg4" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js" integrity="sha384-mll67QQFJfxn0IYznZYonOWZ644AWYC+Pt2cHqMaRhXVrursRwvLnLaebdGIlYNa" crossorigin="anonymous"></script>
    """
    
    # Twikoo 预加载
    twikoo_preload = """
    <link rel="preload" href="https://registry.npmmirror.com/twikoo/1.6.44/files/dist/twikoo.min.js" as="script">
    """
    
    # 将资源添加到页面头部（如果还没有的话）
    if 'katex' not in html and 'twikoo' not in html:
        # 在 </head> 前插入资源
        if '</head>' in html:
            html = html.replace('</head>', katex_resources + twikoo_preload + '</head>')
    
    return html