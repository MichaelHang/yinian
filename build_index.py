# -*- coding: utf-8 -*-
"""
构建诗词语义向量索引（在 Mac 上运行）

把 poetry.db 中每首诗编码成向量，供 search.py 做"意境搜索"。
- 语料文本 = 《标题》朝代·作者 [诗体] + 正文（按句切分、超长诗分块）
- 向量归一化后存为 float16，检索时用点积即余弦相似度

用法：
  正式构建（推荐模型，M2 Max 约 1~2 小时）:
    python build_index.py --model BAAI/bge-large-zh-v1.5
  先小试:
    python build_index.py --limit 3000 --model BAAI/bge-small-zh-v1.5
"""
import argparse
import json
import os
import re
import sqlite3
import time

import numpy as np
from sentence_transformers import SentenceTransformer

# 单块文本上限（字符）。BGE 中文模型上限 512 token，约对应 400+ 汉字，
# 留出标题/作者/诗体的前缀余量。
CHUNK_MAX_CHARS = 380
MAX_CHUNKS_PER_POEM = 12


def load_poems(db_path, limit=None, sample=None, famous_ids=None):
    con = sqlite3.connect(db_path)
    where = """
        WHERE (p.content_hash IS NULL
           OR p.id IN (SELECT MIN(id) FROM poems_zh_hans
                       WHERE content_hash IS NOT NULL GROUP BY content_hash))
    """
    if famous_ids:
        marks = ",".join("?" * len(famous_ids))
        where += f"  AND p.id IN ({marks})"
    sql = f"""
        SELECT p.id, p.title, p.content, a.name, d.name, t.name
        FROM poems_zh_hans p
        LEFT JOIN authors_zh_hans a    ON a.id = p.author_id
        LEFT JOIN dynasties_zh_hans d ON d.id = p.dynasty_id
        LEFT JOIN poetry_types_zh_hans t ON t.id = p.type_id
        {where}
    """
    if sample:
        # 全库按比例随机抽样，用于演示索引（前 LIMIT 首会恰好集中在一个诗体）
        sql += f" ORDER BY RANDOM() LIMIT {int(sample)}"
    elif limit:
        sql += f" LIMIT {int(limit)}"
    params = list(famous_ids) if famous_ids else []
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows


def parse_lines(content):
    """content 列是 JSON 数组字符串，如 ["朝辞白帝彩云间，...", "..."]"""
    try:
        lines = json.loads(content)
        if isinstance(lines, list):
            return "\n".join(str(x) for x in lines)
    except (ValueError, TypeError):
        pass
    return content


def split_body(body, max_len=CHUNK_MAX_CHARS):
    """按句末标点切分成不超过 max_len 的块，避免把一句诗拦腰截断"""
    if len(body) <= max_len:
        return [body]
    sentences = [s for s in re.split(r"(?<=[。！？；!?;\n])", body) if s]
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > max_len:
            chunks.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        chunks.append(cur)
    # 极少数超长诗最多取前 12 块，防止索引无限膨胀
    return chunks[:MAX_CHUNKS_PER_POEM]


def build_corpus(rows):
    """返回 (texts, metas)，每个元素对应一个向量"""
    texts, metas = [], []
    for poem_id, title, content, author, dynasty, ptype in rows:
        title = title or "无题"
        author = author or "佚名"
        dynasty = dynasty or ""
        ptype = ptype or "诗"
        head = f"《{title}》{dynasty}·{author} [{ptype}]\n"
        body = parse_lines(content)
        for chunk in split_body(body):
            texts.append(head + chunk)
            metas.append({
                "poem_id": poem_id,
                "title": title,
                "author": author,
                "dynasty": dynasty,
                "ptype": ptype,
            })
    return texts, metas


def main():
    ap = argparse.ArgumentParser(description="构建诗词语义向量索引")
    ap.add_argument("--db", default="poetry.db")
    ap.add_argument("--out", default="index")
    ap.add_argument("--model", default="BAAI/bge-base-zh-v1.5",
                    help="嵌入模型，正式推荐 BAAI/bge-large-zh-v1.5")
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 首诗（试跑用）")
    ap.add_argument("--sample", type=int, default=None, help="全库随机抽 N 首（演示索引用）")
    ap.add_argument("--famous", default=None, help="只索引名篇清单（make_famous.py 生成的 json）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="留空自动；Apple 芯片用 mps")
    args = ap.parse_args()

    t0 = time.time()
    famous_ids = None
    if args.famous:
        with open(args.famous, encoding="utf-8") as f:
            famous_ids = [x["poem_id"] for x in json.load(f)]
        print(f"名篇清单: {len(famous_ids)} 首")
    rows = load_poems(args.db, args.limit, args.sample, famous_ids)
    texts, metas = build_corpus(rows)
    print(f"载入 {len(rows)} 首诗，切分为 {len(texts)} 个向量块")

    model = SentenceTransformer(args.model, device=args.device)
    if hasattr(model, "max_seq_length"):
        # BGE zh-v1.5 系列上限 512 token；bge-m3 支持 8192，
        # 取 1024 足够覆盖"标题/作者/诗体前缀 + 380 字切块"不被截断
        model.max_seq_length = 512 if "zh-v1.5" in args.model else 1024
    print(f"模型 {args.model} 加载完毕（device={model.device}），开始编码...")

    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)

    os.makedirs(args.out, exist_ok=True)
    # 先写临时文件再原子替换，避免读方（如运行中的服务 mmap）读到半成品
    # 注意 np.save 会给不以 .npy 结尾的名字自动补后缀，故临时文件命名为 *.tmp.npy
    tmp_path = os.path.join(args.out, "vectors.tmp.npy")
    np.save(tmp_path, vectors)
    os.replace(tmp_path, os.path.join(args.out, "vectors.npy"))
    with open(os.path.join(args.out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "num_vectors": len(texts),
                   "num_poems": len(rows)}, f, ensure_ascii=False)

    dt = time.time() - t0
    print(f"完成：{len(texts)} 条向量写入 {args.out}/，"
          f"维度 {vectors.shape[1]}，耗时 {dt/60:.1f} 分钟")


if __name__ == "__main__":
    main()
