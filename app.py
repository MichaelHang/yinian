# -*- coding: utf-8 -*-
"""
意境搜诗 Web 服务

启动（演示索引）：
  python app.py --index index_demo
启动（全量索引，Mac 上）：
  python app.py --index index --device mps

"AI 读心"用的大模型改写，二选一：
  在线 API（任一 OpenAI 兼容服务，推荐）：
    export LLM_API_BASE=https://open.bigmodel.cn/api/paas/v4   # 智谱，示例
    export LLM_API_KEY=你的密钥
    export LLM_MODEL=glm-4-flash
  或本机 Ollama：安装后 `ollama pull qwen3:8b` 即可，无需其他配置。
  都不配置时，"AI 读心"会提示不可用并自动降级为直接检索。
页面：http://127.0.0.1:8000
"""
import argparse
import json
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import search as engine

_state = {}

LLM_CONFIG_FILE = "llm_config.json"


class LLMConfig(BaseModel):
    api_base: str = ""
    api_key: str = ""
    model: str = ""


def _load_llm_config():
    try:
        with open(LLM_CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _effective_llm():
    """界面保存的配置优先，其次环境变量/命令行参数"""
    ui = _state.get("llm_ui") or {}
    if ui.get("api_base") and ui.get("api_key") and ui.get("model"):
        return ui
    return _state.get("llm")


@asynccontextmanager
async def lifespan(app):
    vectors, metas, info = engine.load_index(_state["args"].index)
    # 编码模式：embedding_config.json 的 provider 决定是否加载本地模型
    # （provider=remote 时模型不加载，内存省 ~3GB，查询编码走远端 API）
    ecfg = engine.load_embed_config()
    _state["embed_cfg"] = ecfg
    if ecfg.get('provider') != 'remote':
        from sentence_transformers import SentenceTransformer
        _state["model"] = SentenceTransformer(ecfg.get('local', {}).get('model_path', info["model"]),
                                              device=_state["args"].device)
    _state.update(
        vectors=vectors,
        metas=metas,
        info=info,
    )
    tiers = engine.load_tiers(_state["args"].db)
    _state["chunk_tiers"] = engine.chunk_tiers_for(metas, tiers)
    yield
    _state.clear()


app = FastAPI(title="一念 · 意境搜诗", lifespan=lifespan)

# CORS：前端（Cloudflare Pages）与 API 不同域时浏览器需要
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yinian.dailyai.space",       # Pages 前端（正式域名）
        "https://yinian-3co.pages.dev",       # Pages 默认域名（过渡期）
        "http://127.0.0.1:8000",              # 本地开发
        "http://127.0.0.1:8123",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")


def parse_lines(content):
    try:
        lines = json.loads(content)
        if isinstance(lines, list):
            return [str(x) for x in lines]
    except (ValueError, TypeError):
        pass
    return [content]


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/api/meta")
def meta():
    return {"num_poems": _state["info"]["num_poems"],
            "num_vectors": _state["info"]["num_vectors"],
            "model": _state["info"]["model"]}


@app.get("/api/llm")
def get_llm():
    eff = _effective_llm() or {}
    return {"configured": bool(eff),
            "api_base": eff.get("api_base") or "",
            "model": eff.get("model") or "",
            "has_key": bool(eff.get("api_key"))}


@app.post("/api/llm")
def set_llm(cfg: LLMConfig):
    # 三项全部留空 = 清除配置；仅密钥留空 = 沿用已保存的密钥（页面不回显密钥）
    stored = _state.get("llm_ui") or {}
    api_base, model, api_key_submitted = cfg.api_base.strip(), cfg.model.strip(), cfg.api_key.strip()
    if not api_base and not model and not api_key_submitted:
        _state["llm_ui"] = None
        try:
            os.remove(LLM_CONFIG_FILE)
        except OSError:
            pass
        return get_llm()
    api_key = api_key_submitted or stored.get("api_key", "")
    if not (api_base and model and api_key):
        raise HTTPException(400, "API 地址、密钥、模型名需填写完整（密钥已保存过可留空；清除请三项全部留空）")
    _state["llm_ui"] = {"api_base": api_base, "api_key": api_key, "model": model}
    with open(LLM_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(_state["llm_ui"], f, ensure_ascii=False)
    return get_llm()


@app.post("/api/llm/test")
def test_llm(cfg: LLMConfig):
    stored = _state.get("llm_ui") or {}
    api_base = cfg.api_base.strip() or stored.get("api_base", "")
    api_key = cfg.api_key.strip() or stored.get("api_key", "")
    model = cfg.model.strip() or stored.get("model", "")
    if not (api_base and api_key and model):
        raise HTTPException(400, "请先填写完整的 API 地址、密钥、模型名")
    try:
        text = engine.api_rewrite("今天很累，想找首诗慰劳自己", api_base, api_key, model)
        return {"ok": True, "sample": text[:80]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/search")
def api_search(q: str = Query(min_length=1, max_length=200),
               top: int = Query(8, ge=1, le=20),
               ai: bool = False):  # V1：原句直搜；V2 上读心时改回 True
    text, rewritten, ai_error = q, None, None
    if ai:
        llm = _effective_llm()
        if not llm:
            ai_error = "尚未配置大模型：点页面上的「⚙ 设置」填入 API，或本机运行 Ollama"
        else:
            try:
                rewritten = engine.rewrite_query(q, llm)
                if rewritten:
                    text = rewritten
            except Exception:
                ai_error = "大模型改写失败（检查 API 地址/密钥，或 Ollama 是否在运行），已按原文检索"

    # 查询编码：按 embedding_config 分发（local / remote）
    ecfg = _state["embed_cfg"]
    if ecfg.get('provider') == 'remote':
        try:
            vec = np.array(engine.encode_remote(text, ecfg['remote']), dtype=np.float32)
        except Exception:
            # remote 失败：本地模型可用才回落，纯 remote 部署则明确报错
            import os as _os
            local_path = ecfg.get('local', {}).get('model_path', '')
            if _os.path.isdir(local_path):
                if 'model' not in _state:
                    from sentence_transformers import SentenceTransformer
                    _state["model"] = SentenceTransformer(local_path,
                                                          device=_state["args"].device)
                vec = engine.encode_query_fallback(text, _state["info"]["model"], ecfg)
            else:
                raise HTTPException(503, "检索服务暂不可用（远端编码失败且本地模型未部署）")
    else:
        prefix = engine.QUERY_INSTRUCTION if engine.needs_instruction(ecfg.get('local', {}).get('model_path', _state["info"]["model"])) else ""
        vec = engine.encode_query_model(ecfg.get('local', {}).get('model_path', _state["info"]["model"]),
                                        prefix + text, embed_cfg=ecfg)

    qboost = engine.quote_boost_for(
        _state["metas"], engine.fts_quote_ids(_state["args"].db, q))
    candidates = engine.semantic_top(_state["vectors"], _state["metas"], vec, top,
                                     _state.get("chunk_tiers"),
                                     _state["args"].tier_weight, qboost)
    poems = engine.fetch_poems(_state["args"].db, [m["poem_id"] for m, _ in candidates])

    results = []
    for m, score in candidates:
        _, content = poems[m["poem_id"]]
        results.append({
            "title": m["title"], "author": m["author"],
            "dynasty": m["dynasty"], "ptype": m["ptype"],
            "lines": parse_lines(content), "score": round(score, 4),
        })
    return {"query": q, "rewritten": rewritten, "ai_error": ai_error,
            "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="poetry.db")
    ap.add_argument("--index", default="index_demo")
    ap.add_argument("--device", default=None, help="Apple 芯片建议 mps")
    ap.add_argument("--tier-weight", type=float, default=0.03,
                    help="名气档位加权系数（0 关闭；tier 0~3，相似度 + w*tier）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--llm-model", default=None, help="在线模型名或 Ollama 模型名")
    ap.add_argument("--llm-api-base", default=None, help="OpenAI 兼容 API 地址")
    ap.add_argument("--llm-api-key", default=None)
    _state["args"] = ap.parse_args()

    llm = {
        "api_base": _state["args"].llm_api_base or os.getenv("LLM_API_BASE"),
        "api_key": _state["args"].llm_api_key or os.getenv("LLM_API_KEY"),
        "model": _state["args"].llm_model or os.getenv("LLM_MODEL"),
    }
    _state["llm"] = llm if any(llm.values()) else None
    _state["llm_ui"] = _load_llm_config()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=_state["args"].port)
