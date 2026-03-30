// docs/javascripts/gitalk-init.js
document.addEventListener('DOMContentLoaded', function() {
    // 等待页面完全加载
    setTimeout(initGitalk, 1000);
});

function initGitalk() {
    // 检查是否已经存在 Gitalk 容器
    if (document.getElementById('gitalk-container')) {
        return;
    }
    
    // 获取当前页面信息
    const pageTitle = document.title;
    const pageUrl = window.location.href;
    
    // 创建 Gitalk 容器
    const container = document.createElement('div');
    container.id = 'gitalk-container';
    container.style.marginTop = '2rem';
    container.style.padding = '1rem 0';
    container.style.borderTop = '1px solid var(--md-default-fg-color--lightest)';
    
    // 查找合适的位置插入评论框
    const content = document.querySelector('.md-content');
    if (content) {
        content.appendChild(container);
        
        // 初始化 Gitalk
        const gitalk = new Gitalk({
            clientID: '你的 GitHub Application Client ID',
            clientSecret: '你的 GitHub Application Client Secret',
            repo: 'Mathematical-Foundations-of-Reinforcement-Learning-Notes', // 仓库名
            owner: 'wgyhhhh', // GitHub 用户名
            admin: ['wgyhhhh'], // 管理员列表
            id: generateGitalkId(), // 确保唯一性
            title: pageTitle,
            body: `页面: ${pageTitle}\nURL: ${pageUrl}`,
            language: 'zh-CN',
            distractionFreeMode: false,
            createIssueManually: true, // 如果页面没有对应的 issue 则自动创建
        });
        
        try {
            gitalk.render('gitalk-container');
        } catch (error) {
            console.log('Gitalk 渲染失败:', error);
        }
    }
}

// 生成唯一的 ID（使用页面路径）
function generateGitalkId() {
    const path = window.location.pathname;
    // 移除开头的斜杠和文件扩展名
    return path.replace(/^\//, '').replace(/\.md$/, '').replace(/\//g, '-') || 'home';
}