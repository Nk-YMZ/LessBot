# LessBot 🤖

> 基于 "Less is More" 核心思想，追求最真实聊天体验的极简赛博生命。

LessBot 抛弃了传统 QQ 机器人臃肿的插件生态和刻板的指令系统，专注于 **“拟人化”** 与 **“情绪价值”** 。通过 **非对称记忆隔离架构** ，旨在解决大语言模型（LLM）在长期群聊中容易出现的“自我洗脑”、“烂梗缝合”和“话题脱节”问题。

## 🌟 核心特性 (Core Features)

* **🎭 双轨模型架构 (Director-Actor Pipeline):** 
    * **导演模型 (The Brain):** 拥有全局 50 条完整记忆，负责捕捉群聊最新话题、感知整体氛围，并向演员下达指令。
    * **演员模型 (The Mouth):** 仅拥有极短的纯净上下文（物理屏蔽自身历史发言），严格按照导演指令进行演绎，尽可能避免 AI 的“烂梗复读”幻觉。
* **⏱️ 滑动窗口防抖 (Debounce Buffer):** 独创基于时间窗口的消息合并机制（默认 5 秒），群友停止高频输出后才会触发思考，不做“赛博话痨”。
* **🧠 多线程记忆隔离:** 基于哈希表的群组状态锁，支持同时挂载多个群聊，不同群聊之间的记忆与人格实现物理级隔离。
* **⌨️ 物理层真实拟态:** 动态模拟真人打字延迟，支持长句智能分段发送，去除一切 AI 痕迹。
* **⚡ 极简底层依赖:** 基于 Python原生 `asyncio`，底层对接 [NapCatQQ](https://github.com/NapNeko/NapCatQQ) 的 WebSocket 接口。**无任何臃肿机器人框架**，开箱即用。

## 🚀 快速开始 (Quick Start)

### 1. 部署前置基建 (NapCatQQ)
本项目底层依赖 NapCatQQ 提供的无头服务。推荐使用 Docker 部署并暴露 `3001` WebSocket 端口。

### 2. 克隆与环境准备
确保系统安装有 Python 3.10+ 环境。
```bash
git clone [https://github.com/Nk-YMZ/LessBot.git](https://github.com/Nk-YMZ/LessBot.git)
cd LessBot
python -m venv .venv
source .venv/bin/activate
pip install websockets httpx
```

### 3. 配置
1. 将 `template/` 目录下的所有 `.json` 文件复制一份并去除后缀到项目根目录。
2. 配置 `models_config.json`: 填入你的大模型 API Key（默认适配 DeepSeek）。
3. 配置 `main_config.json`: 设定你想要接管的群聊白名单 (`allowed_groups`)。

### 4. 运行
```bash
# 前台测试运行
python main.py

# 若要使其后台常驻，麻烦您自己解决一下
```

## 📝 架构设计图 (简述)
`群消息` -> `滑动窗口缓冲池` -> `3秒防抖结束` -> `提取 50 条全局记忆` -> `导演模型提炼氛围与话题` -> `构建 3 条纯净切片` -> `演员模型执行生成` -> `分段打字发送` -> `记忆烙印`

---
> 🤖 **声明 (Disclaimer):** 本项目的核心架构思路与产品定义由苦逼大学生提供，具体代码实现与重构系由 AI 辅助完成。赞美钛君！
*Built with ❤️ & Arch Linux.*