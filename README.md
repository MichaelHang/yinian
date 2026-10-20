# 一念 · 意境搜诗

用户输入一句现代白话（如"深夜想家了"），按**意境/情感**从古诗词库中检索最贴近的诗。
水墨古风单页 + FastAPI。检索 = 语义向量 + 名气档位加权 + 引句精确匹配（三信号）。

## 当前状态

- **库**：poetry_full.db 322,158 首（全量，Mac 主力）/ poetry_lite.db（部署候选，范围随选型）
- **打分**：100% 覆盖（LLM 0-10 分 + 选本标记，已经完成）
- **检索**：三信号排序（w=0.03 定稿）；FTS5 引句精确通道（引用原句必中出处）
- **AI 读心**：已启用——LLM 配置经环境变量

## 快速开始（Mac 本地）

```bash
# 1. 环境
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# 2. 模型：models/bge-m3/（约 4.3GB；新机器需先下载，已有可跳过）
#    国内网络加前缀 HF_ENDPOINT=https://hf-mirror.com 走镜像
.venv/bin/huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
# 3. （首次运行或索引缺失时）建索引：poetry_lite.db → index_lite/（M 系列 Mac 约 2~3 分钟）
HF_HUB_OFFLINE=1 .venv/bin/python build_index.py \
    --db poetry_lite.db --model models/bge-m3 --device mps --out index_lite
```

**两个可选配置**（都不创建也能跑，起服务前按需创建）：

- `embedding_config.json`——查询编码模式。**缺省即 local**（本地模型编码）；
  想走远端编码（省本机内存，首次查询有网络往返）时创建：

  ```json
  { "provider": "remote",
    "local":  { "model_path": "models/bge-m3" },
    "remote": { "provider_id": "siliconflow",
                "api_base": "https://api.siliconflow.cn/v1",
                "api_key": "sk-…",
                "model": "BAAI/bge-m3" } }
  ```

- `llm_config.json`——AI 读心（缺省不启用，前端自动降级为原句检索，不报错）。
  任一 OpenAI 兼容服务均可；三项填全才生效，优先级高于环境变量：

  ```json
  { "api_base": "https://open.bigmodel.cn/api/paas/v4",
    "api_key": "sk-…",
    "model": "glm-4-flash" }
  ```

```bash
# 4. 起服务（lite 库 + 本地编码）
.venv/bin/python app.py --index index_lite --db poetry_lite.db --device mps --port 8123
# 浏览器 http://127.0.0.1:8123（调试视图加 ?debug=1）
```

## 部署到自己的服务器（前后端一体）

一台服务器同时提供页面与检索 API：FastAPI 自带页面，前后端同源，无需跨域配置、
无需向量数据库、无需前端构建。查询编码走 SiliconFlow API（remote 模式），
服务器不用装深度学习环境，**1GB 内存即可**。

**准备三样**：

1. 一台 Linux 服务器（1GB 内存起步，推荐 Ubuntu/Debian）；
2. 一个域名，A 记录解析到服务器 IP（不解析可先用 IP 访问，但没有 HTTPS）；
3. 一个 SiliconFlow API key（查询编码用，免费注册可得）。

**需要的文件**（Mac 项目目录里的 5 项）：

| 文件 | 作用 | 备注 |
|---|---|---|
| `app.py` / `search.py` | 服务与检索引擎 | 原样上传 |
| `static/` | 前端页面与站点图标 | 原样上传 |
| `poetry_lite.db` | 诗库（15,505 首） | 原样上传 |
| `index_lite/` | 向量索引三件套 | 原样上传 |
| `embedding_config.json` | 编码配置 | 不上传，服务器上新建（provider 改 remote + 填 key） |

**步骤总览**（每步的具体命令见 **DEPLOY_STEPS.md**，编号一一对应）：

1. 服务器装系统依赖（Python 虚拟环境 + Caddy + 防火墙放行）；
2. 上传部署文件到服务器 `/opt/yinian` 目录；
3. 在服务器上创建编码配置（provider=remote、填入 SiliconFlow key）；
4. 建 Python 虚拟环境、装依赖、试运行验证；
5. 注册 systemd 服务常驻；
6. Caddy 反向代理绑定域名（自动 HTTPS）；
7. 验证线上访问；
8. （可选）启用 AI 读心：在服务器上用环境变量写死大模型配置，无需界面设置。

想用全量库（32.2 万首）内存不变、磁盘多占约 1GB。

## 原理

1. 每首诗按句切块，用 bge-m3 编码成 1024 维向量（归一化，float16 存
   `index_*/vectors.npy`），块元信息存 `chunks.jsonl`；
2. 用户查询同样编码成向量，与全库向量做矩阵乘（余弦相似度）；
3. 排序 = 相似度 + 名气档位加权（tier）+ 引句精确匹配（FTS5 trigram）；
4. 命中诗的正文从 SQLite 取出返回。

**索引（index_lite / index_full）怎么产生**：在 Mac 上用 `build_index.py` 构建
（需要 MPS + 本地模型 + sentence-transformers），服务器只读取不构建：

```bash
# 精简版：poetry_lite.db → index_lite/（15,505 首，MPS 约 2 分钟）
HF_HUB_OFFLINE=1 .venv/bin/python build_index.py \
    --db poetry_lite.db --model models/bge-m3 --device mps --out index_lite

# 全量版：poetry_full.db → index_full/（322,158 首，约 40 分钟）
HF_HUB_OFFLINE=1 .venv/bin/python build_index.py \
    --db poetry_full.db --model models/bge-m3 --device mps --out index_full
```

## 已知限制

- 语料只含诗词曲，不含文赋（《滕王阁序》等搜不到；扩展方案已调研存档）；
- 《南吕·一枝花不伏老》等散曲套数在语料中是节选；
- 跨作者同文异文约 985 组（归属公案类），维持现状逐条处理；
