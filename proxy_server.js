const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 8081; // 与你之前的 Nginx 端口保持一致，方便不用改前端配置

// 1. 启用全局 CORS，允许任何来源访问 (解决 Failed to fetch)
app.use(cors());

// 2. 静态文件服务
// 访问 http://localhost:8081 即可看到 index.html
app.use(express.static(path.join(__dirname, './')));

// 3. 代理配置
const targetUrl = 'https://ie0wp9c4uz-8300.cnb.run'; // 你的远程 API 地址

const proxyOptions = {
    target: targetUrl,
    changeOrigin: true, // 修改 Host 头，让远程服务器以为是本地请求
    secure: false,      // 如果远程是自签名证书，设为 false
    ws: true,           // 支持 WebSocket
    // 关键：处理响应头，强制允许跨域
    onProxyRes: function (proxyRes, req, res) {
        proxyRes.headers['Access-Control-Allow-Origin'] = '*';
        proxyRes.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS';
        proxyRes.headers['Access-Control-Allow-Headers'] = '*';
        // 移除可能导致冲突的头
        delete proxyRes.headers['access-control-allow-origin']; 
        // 重新加上我们自己的
        res.setHeader('Access-Control-Allow-Origin', '*');
    },
    onError: (err, req, res) => {
        console.error('代理错误:', err);
        res.status(500).send('Proxy Error');
    }
};

// 4. 应用代理到 /v1 和 /v2 接口
// 这样前端请求 http://localhost:8081/v1/xxx 就会被转发到远程
app.use('/v1', createProxyMiddleware(proxyOptions));
app.use('/v2', createProxyMiddleware(proxyOptions));

// 启动服务
app.listen(PORT, () => {
    console.log(`\n🚀 本地代理服务器已启动!`);
    console.log(`👉 访问地址: http://localhost:${PORT}`);
    console.log(`🔧 API 代理: /v1, /v2 -> ${targetUrl}`);
    console.log(`📂 静态文件: ${__dirname}\n`);
});
