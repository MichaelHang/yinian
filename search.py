# -*- coding: utf-8 -*-
"""
意境搜诗：输入一句现代白话，按语义（意境/情感）找出最贴近的诗词。

用法：
  python search.py "最近有点迷茫，看不清前路"
  python search.py "..." --top 10
  python search.py "..." --llm                 # 先用 Ollama 本地大模型把白话转成古典意境描述再检索（效果更好）
  python search.py "..." --rerank              # 用 bge-reranker 对候选精排（首次需额外下载约 2GB 模型）

依赖：build_index.py 生成的 index/ 目录、poetry.db；
LLM 改写需先 `ollama pull qwen3:8b` 并保持 ollama 运行。
"""
import argparse
import json
import os
import re
import sqlite3
import time
import urllib.request

import numpy as np

# BGE 中文 v1.5 官方建议：短查询前加此指令，可显著提升检索效果。
# 注意：bge-m3 官方明确不需要指令前缀，硬套反而伤害效果，故按模型条件使用。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def needs_instruction(model_name):
    """BGE zh-v1.5 系列需要查询指令；bge-m3 等其余模型不需要"""
    return "zh-v1.5" in (model_name or "")


# ── 查询编码：local / remote 双模式（embedding_config.json 配置驱动）──
ECFG_FILE = 'embedding_config.json'
_ef_cache = {'model': None}


def load_embed_config():
    try:
        return json.load(open(ECFG_FILE, encoding='utf-8'))
    except (OSError, ValueError):
        return {'provider': 'local', 'local': {'model_path': 'models/bge-m3'}}


def encode_remote(text, rcfg):
    """远端 embedding（OpenAI 兼容 /embeddings 接口），返回归一化向量 list"""
    import time as _t
    body = {"model": rcfg['model'], "input": [text], "encoding_format": "float"}
    last = None
    for i in range(3):
        req = urllib.request.Request(
            rcfg['api_base'].rstrip('/') + '/embeddings',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json',
                     'Authorization': f"Bearer {rcfg['api_key']}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read())
            vec = d['data'][0]['embedding']
            norm = sum(x * x for x in vec) ** 0.5
            return [x / norm for x in vec]
        except Exception as e:
            last = e
            if i + 1 < 3:
                _t.sleep(2)
    raise RuntimeError(f'远端 embedding 失败（{rcfg.get("provider_id")}）: {last}')


def encode_query_model(model_name, text, use_instruction=None, embed_cfg=None):
    """按配置分发：provider=remote 走远端 API；否则本地 SentenceTransformer"""
    cfg = embed_cfg if embed_cfg is not None else load_embed_config()
    if cfg.get('provider') == 'remote':
        return encode_remote(text, cfg['remote'])
    # 本地模式（原逻辑）
    if use_instruction is None:
        use_instruction = needs_instruction(model_name)
    if use_instruction:
        text = QUERY_INSTRUCTION + text
    if _ef_cache['model'] is None or _ef_cache['model'][0] != model_name:
        from sentence_transformers import SentenceTransformer
        _ef_cache['model'] = (model_name, SentenceTransformer(model_name))
    v = _ef_cache['model'][1].encode([text], normalize_embeddings=True,
                                     convert_to_numpy=True)[0]
    return v.astype(np.float32)


def encode_query_fallback(text, model_name, embed_cfg):
    """remote 失败时的本地兜底（本地模型可用才成功）"""
    if use_instruction := needs_instruction(model_name):
        text = QUERY_INSTRUCTION + text
    if _ef_cache['model'] is None or _ef_cache['model'][0] != model_name:
        from sentence_transformers import SentenceTransformer
        _ef_cache['model'] = (model_name, SentenceTransformer(model_name))
    return _ef_cache['model'][1].encode([text], normalize_embeddings=True,
                                        convert_to_numpy=True)[0].astype(np.float32)

REWRITE_PROMPT = (
    "你是诗词检索助手。用户会用现代白话描述自己的处境或心情，"
    "请把它转化为适合在古典诗词库中做语义检索的查询描述。要求：\n"
    "1. 概括核心情感（如：失意、落寞、怀才不遇、羁旅孤寂）；\n"
    "2. 点出古典诗词中表达这种心境的典型意象（如：秋风、孤灯、寒江、白发、浊酒）；\n"
    "3. 最后用一段30~60字的古典意境描述收尾，不要出现任何现代词汇。\n"
    "只输出这段检索文本，不要解释。\n"
    "用户输入：{q}"
)


def load_index(index_dir):
    vectors = np.load(f"{index_dir}/vectors.npy", mmap_mode="r")
    metas = []
    with open(f"{index_dir}/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    with open(f"{index_dir}/meta.json", encoding="utf-8") as f:
        info = json.load(f)
    print(f"索引：{info['num_vectors']} 条向量（{info['num_poems']} 首诗），模型 {info['model']}")
    return vectors, metas, info


def encode_query(model_name, text, use_instruction=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    if use_instruction is None:
        use_instruction = needs_instruction(model_name)
    if use_instruction:
        text = QUERY_INSTRUCTION + text
    v = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return v.astype(np.float32)


def ollama_rewrite(query, model="qwen3:8b"):
    """用本地 Ollama 大模型改写查询（弥合古今语域差异）"""
    prompt = REWRITE_PROMPT.format(q=query)
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    text = data["message"]["content"].strip()
    print(f"\n[LLM 改写 → {model}]\n{text}\n")
    return text


def api_rewrite(query, base_url, api_key, model, attempts=3):
    """用 OpenAI 兼容的在线大模型改写查询（智谱/DeepSeek/通义/SiliconFlow 等均适用）
    免费档服务延迟方差大，思考型模型改写常需 30-60s，超时放宽并多重试"""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": REWRITE_PROMPT.format(q=query)}],
        "temperature": 0.2,
        "stream": False,
    }
    data = None
    last_err = Exception("未知错误")
    for i in range(attempts):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            break
        except Exception as e:
            last_err = e
            if i + 1 < attempts:
                time.sleep(1)
    if data is None:
        raise last_err
    text = data["choices"][0]["message"]["content"].strip()
    # 兼容 qwen3 / deepseek-r1 等思考型模型：去掉思考段
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    print(f"\n[LLM 改写 → {model}]\n{text}\n")
    return text


def rewrite_query(query, llm):
    """按配置选择改写后端：llm 为 None/空 → 原样返回；含 api 配置 → 在线；否则 Ollama"""
    if not llm:
        return None
    if llm.get("api_base") and llm.get("api_key") and llm.get("model"):
        return api_rewrite(query, llm["api_base"], llm["api_key"], llm["model"])
    return ollama_rewrite(query, llm.get("model") or "qwen3:8b")


def fetch_poems(db_path, poem_ids):
    con = sqlite3.connect(db_path)
    marks = ",".join("?" * len(poem_ids))
    rows = con.execute(
        f"SELECT id, title, content FROM poems_zh_hans WHERE id IN ({marks})",
        list(poem_ids),
    ).fetchall()
    con.close()
    return {r[0]: (r[1], r[2]) for r in rows}


def display(meta, content, score):
    try:
        lines = json.loads(content)
        if not isinstance(lines, list):
            raise ValueError
        body = "\n".join(str(x) for x in lines)
    except (ValueError, TypeError):
        body = content
    print(f"\n{'─' * 56}")
    print(f"{meta['ptype']} | {meta['dynasty']}·{meta['author']} 《{meta['title']}》  相似度 {score:.4f}")
    print(body)


def load_tiers(db_path):
    """读 tier 列（poem_id → tier），无 tier 列时返回 None"""
    try:
        con = sqlite3.connect(db_path)
        rows = con.execute("SELECT id, tier FROM poems_zh_hans WHERE tier IS NOT NULL").fetchall()
        con.close()
        if not rows:
            return None
        max_id = max(r[0] for r in rows)
        tiers = np.zeros(max_id + 1, dtype=np.float32)
        for pid, t in rows:
            tiers[pid] = t
        return tiers
    except sqlite3.Error:
        return None


def chunk_tiers_for(metas, tiers):
    """把 poem 级 tier 对齐到 chunk 顺序"""
    if tiers is None:
        return None
    return np.array([tiers[m["poem_id"]] if m["poem_id"] < len(tiers) else 0.0
                     for m in metas], dtype=np.float32)


def fts_quote_ids(db_path, query, min_len=4):
    """精确引句检测：FTS5 trigram 短语匹配，返回原文命中 query 的 poem_id 集合。
    稠密向量对"逐字引用"有盲区（整段语义稀释原句），用词面精确匹配补位。"""
    q = (query or '').strip()
    if len(q) < min_len:
        return set()
    candidates = [q]
    if re.search(r'[，。！？；、,!?;]', q):
        candidates += [seg for seg in re.split(r'[，。！？；、,!?;]', q)
                       if len(seg) >= min_len]
    ids = set()
    try:
        con = sqlite3.connect(db_path)
        for cand in candidates:
            safe = cand.replace('"', '""')
            try:
                rows = con.execute(
                    "SELECT rowid FROM poems_fts WHERE poems_fts MATCH ? LIMIT 50",
                    (f'"{safe}"',)).fetchall()
                ids.update(r[0] for r in rows)
            except sqlite3.OperationalError:
                pass
        con.close()
    except sqlite3.Error:
        return set()
    return ids


def quote_boost_for(metas, fts_ids, boost=0.30):
    """把引句命中的 poem_id 转成 chunk 级加分阵（None 表示无命中）。
    +0.30：引用原句是极强的意图信号，需盖过题材相近诗的语义分差
    （无名诗稠密分可低至 0.45，+0.15 会被 0.68 的主题诗压住）"""
    if not fts_ids:
        return None
    return np.array([boost if m["poem_id"] in fts_ids else 0.0
                     for m in metas], dtype=np.float32)


def semantic_top(vectors, metas, qvec, top, chunk_tiers=None, tier_weight=0.0,
                 quote_boost=None):
    """向量召回 + tier 加权 + 引句精确匹配加权 + 按诗去重，返回 [(meta, score), ...]"""
    scores = vectors.astype(np.float32) @ qvec
    if chunk_tiers is not None and tier_weight:
        scores = scores + tier_weight * chunk_tiers
    if quote_boost is not None:
        scores = scores + quote_boost
    order = np.argsort(-scores)[: top * 40]
    best, meta_by_poem = {}, {}
    for i in order:
        pid = metas[i]["poem_id"]
        if pid not in best or scores[i] > best[pid]:
            best[pid] = float(scores[i])
            meta_by_poem[pid] = metas[i]
        if len(best) >= top * 4:
            break
    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:top]
    return [(meta_by_poem[pid], score) for pid, score in ranked]


def main():
    ap = argparse.ArgumentParser(description="意境搜诗")
    ap.add_argument("query", help="你的现代白话描述，如：最近有点迷茫，看不清前路")
    ap.add_argument("--db", default="poetry.db")
    ap.add_argument("--index", default="index")
    ap.add_argument("--model", default=None, help="覆盖索引里记录的嵌入模型")
    ap.add_argument("--top", type=int, default=5, help="返回前 N 首")
    ap.add_argument("--llm", action="store_true", help="先用大模型改写查询（在线 API 或本机 Ollama）")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-api-base", default=None,
                    help="OpenAI 兼容 API 地址，如 https://open.bigmodel.cn/api/paas/v4")
    ap.add_argument("--llm-api-key", default=None)
    ap.add_argument("--rerank", action="store_true", help="用 bge-reranker-v2-m3 精排")
    ap.add_argument("--tier-weight", type=float, default=0.03,
                    help="名气档位加权系数（0 关闭）")
    args = ap.parse_args()

    vectors, metas, info = load_index(args.index)
    ctiers = chunk_tiers_for(metas, load_tiers(args.db))
    model_name = args.model or info["model"]

    query_text = args.query
    if args.llm:
        llm = {
            "api_base": args.llm_api_base or os.getenv("LLM_API_BASE"),
            "api_key": args.llm_api_key or os.getenv("LLM_API_KEY"),
            "model": args.llm_model or os.getenv("LLM_MODEL"),
        }
        query_text = rewrite_query(args.query, llm) or args.query

    q = encode_query(model_name, query_text)
    qboost = quote_boost_for(metas, fts_quote_ids(args.db, args.query))

    # 向量已归一化，点积即余弦相似度。几十万 x 一千维以内，直接矩阵乘即可。
    candidates = semantic_top(vectors, metas, q, args.top, ctiers, args.tier_weight, qboost)
    poems = fetch_poems(args.db, [m["poem_id"] for m, _ in candidates])

    if args.rerank:
        from sentence_transformers import CrossEncoder
        print("\n[bge-reranker-v2-m3 精排中...]")
        cross = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        pairs = [(args.query, poems[m["poem_id"]][1][:600]) for m, _ in candidates]
        rr = cross.predict(pairs)
        candidates = [(m, float(s)) for (m, _), s in sorted(
            zip(candidates, rr), key=lambda t: -t[1])]

    print(f"\n查询：{args.query}\n最贴近意境的 {len(candidates)} 首：")
    for meta, score in candidates:
        _, content = poems[meta["poem_id"]]
        display(meta, content, score)


if __name__ == "__main__":
    main()
