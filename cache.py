import fcntl
import hashlib
import json
import logging
import os
import time

CACHE_FILE = os.path.join(os.path.dirname(__file__), "response_cache.json")

# TTL em segundos por tipo de cache
TTL_DEFAULT = 60 * 60 * 6    # 6 horas — pesquisas gerais
TTL_CVE     = 60 * 60 * 24   # 24 horas — info de CVE muda pouco
TTL_NEWS    = 60 * 60 * 2    # 2 horas — notícias ficam desatualizadas rápido

# Cache em memória para evitar I/O em cada get()
_mem_cache: dict = {}
_mem_loaded: bool = False


def _load() -> dict:
    global _mem_cache, _mem_loaded
    if _mem_loaded:
        return _mem_cache
    if not os.path.exists(CACHE_FILE):
        _mem_cache = {}
        _mem_loaded = True
        return _mem_cache
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                _mem_cache = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        _mem_loaded = True
    except Exception as e:
        logging.error(f"cache._load falhou: {e}")
        _mem_cache = {}
        _mem_loaded = True
    return _mem_cache


def _save(data: dict):
    global _mem_cache
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, ensure_ascii=False)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        _mem_cache = data
    except Exception as e:
        logging.error(f"cache._save falhou: {e}")


def _make_key(*parts: str) -> str:
    combined = "|".join(str(p).lower().strip() for p in parts)
    return hashlib.md5(combined.encode()).hexdigest()


def get(ttl: int, *key_parts: str) -> str | None:
    cache = _load()
    key = _make_key(*key_parts)
    entry = cache.get(key)
    if not entry:
        return None
    # usar o TTL da entrada se disponível, senão o TTL passado como argumento
    entry_ttl = entry.get("ttl", ttl)
    if time.time() - entry["ts"] > entry_ttl:
        cache.pop(key, None)
        _save(cache)
        return None
    return entry["value"]


def set(value: str, *key_parts: str, ttl: int = TTL_DEFAULT):
    cache = _load()
    key = _make_key(*key_parts)
    cache[key] = {"value": value, "ts": time.time(), "ttl": ttl}
    _save(cache)


def purge_expired():
    """Remove todas as entradas expiradas usando o TTL individual de cada entrada."""
    cache = _load()
    now = time.time()
    cleaned = {
        k: v for k, v in cache.items()
        if now - v["ts"] <= v.get("ttl", TTL_CVE)
    }
    if len(cleaned) < len(cache):
        _save(cleaned)
    return len(cache) - len(cleaned)


def stats() -> dict:
    cache = _load()
    now = time.time()
    valid = sum(
        1 for v in cache.values()
        if now - v["ts"] <= v.get("ttl", TTL_CVE)
    )
    return {"total": len(cache), "valid": valid}
