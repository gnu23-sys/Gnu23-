import logging
import time
import requests
import re
import html
import urllib.parse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
}

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 1.5  # segundos base — duplica a cada tentativa


def _get_with_retry(url: str, *, params: dict = None, timeout: int = 10, extra_headers: dict = None) -> requests.Response | None:
    """GET com retry exponencial. Retorna Response ou None se todas as tentativas falharem."""
    headers = {**HEADERS, **(extra_headers or {})}
    delay = _RETRY_BACKOFF
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code < 500:
                return r
            logging.warning(f"_get_with_retry: HTTP {r.status_code} em {url} (tentativa {attempt}/{_RETRY_ATTEMPTS})")
        except (requests.ConnectionError, requests.Timeout) as e:
            logging.warning(f"_get_with_retry: {e} em {url} (tentativa {attempt}/{_RETRY_ATTEMPTS})")
        if attempt < _RETRY_ATTEMPTS:
            time.sleep(delay)
            delay *= 2
    return None


def web_search(query: str, max_results: int = 5) -> str:
    # Tenta DDG HTML scraping primeiro
    result = _ddg_html(query, max_results)
    if result:
        return result
    # Fallback: DDG instant answer API
    return _ddg_api(query, max_results)


def _ddg_html(query: str, max_results: int) -> str:
    try:
        params = {"q": query, "kl": "pt-pt", "kp": "-2"}
        r = _get_with_retry("https://html.duckduckgo.com/html/", params=params, timeout=12)
        if r is None or r.status_code != 200:
            return ""

        # Extrair snippets dos resultados
        pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'
        title_pattern = r'<a class="result__a"[^>]*>(.*?)</a>'
        url_pattern = r'<a class="result__url"[^>]*>(.*?)</a>'

        snippets = re.findall(pattern, r.text, re.DOTALL)
        titles = re.findall(title_pattern, r.text, re.DOTALL)
        urls = re.findall(url_pattern, r.text, re.DOTALL)

        results = []
        for i in range(min(max_results, len(snippets))):
            title = html.unescape(re.sub(r"<[^>]+>", "", titles[i])).strip() if i < len(titles) else ""
            snippet = html.unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip()
            url = urls[i].strip() if i < len(urls) else ""
            if snippet:
                results.append(f"**{title}**\n{snippet}\n{url}")

        return "\n\n---\n\n".join(results) if results else ""
    except Exception as e:
        logging.warning(f"_ddg_html falhou para '{query}': {e}")
        return ""


def _ddg_api(query: str, max_results: int) -> str:
    try:
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = _get_with_retry("https://api.duckduckgo.com/", params=params, timeout=10)
        if r is None:
            return f"Sem resultados para: {query}"
        data = r.json()

        results = []
        if data.get("AbstractText"):
            results.append(f"**{data.get('Heading', '')}**\n{data['AbstractText']}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"{topic['Text']}")

        return "\n\n---\n\n".join(results) if results else f"Sem resultados para: {query}"
    except Exception as e:
        logging.warning(f"_ddg_api falhou para '{query}': {e}")
        return f"Erro na pesquisa: {e}"


def search_cve(cve_id: str) -> str:
    try:
        r = _get_with_retry(f"https://cveawg.mitre.org/api/cve/{cve_id}", timeout=10)
        if r is not None and r.status_code == 200:
            data = r.json()
            cna = data.get("containers", {}).get("cna", {})
            desc = cna.get("descriptions", [{}])[0].get("value", "Sem descrição.")
            metrics = cna.get("metrics", [])
            score = ""
            if metrics:
                cvss = metrics[0].get("cvssV3_1", metrics[0].get("cvssV3_0", {}))
                if cvss:
                    score = f"\nCVSS: {cvss.get('baseScore', 'N/A')} ({cvss.get('baseSeverity', '')})"
            refs = cna.get("references", [])[:3]
            ref_text = "\n".join(f"- {ref.get('url','')}" for ref in refs) if refs else ""
            return f"**{cve_id}**\n{desc}{score}\n\nReferências:\n{ref_text}"
    except Exception as e:
        logging.warning(f"search_cve MITRE API falhou para '{cve_id}': {e}")
    # Fallback web search
    return web_search(f"{cve_id} vulnerability details exploit poc", max_results=4)


def search_tool_help(tool: str, technique: str = "") -> str:
    query = f"{tool} pentest {technique} examples commands".strip()
    return web_search(query, max_results=4)


def search_github_pocs(cve_id: str) -> list[dict]:
    """Pesquisa PoCs no GitHub para um CVE. Retorna lista de dicts com name, url, stars, description."""
    results = []

    # 1. GitHub Search API (sem auth — 10 req/min)
    try:
        api_headers = {**HEADERS, "Accept": "application/vnd.github+json"}
        gh_token = __import__("os").getenv("GITHUB_TOKEN")
        if gh_token:
            api_headers["Authorization"] = f"Bearer {gh_token}"

        r = _get_with_retry(
            "https://api.github.com/search/repositories",
            params={"q": f"{cve_id} poc exploit", "sort": "stars", "per_page": 5},
            timeout=10,
            extra_headers=api_headers,
        )
        if r is not None and r.status_code == 200:
            for item in r.json().get("items", []):
                results.append({
                    "name": item["full_name"],
                    "url": item["html_url"],
                    "stars": item["stargazers_count"],
                    "description": item.get("description") or "",
                    "updated": item.get("updated_at", "")[:10],
                })
    except Exception as e:
        logging.warning(f"search_github_pocs GitHub API falhou para '{cve_id}': {e}")

    # 2. Também pesquisar code/gists — busca web no GitHub como fallback
    if not results:
        try:
            ddg = _ddg_html(f"site:github.com {cve_id} poc exploit", max_results=5)
            urls = re.findall(r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", ddg)
            seen = set()
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    results.append({"name": u, "url": f"https://{u}", "stars": None, "description": "", "updated": ""})
        except Exception as e:
            logging.warning(f"search_github_pocs DDG fallback falhou para '{cve_id}': {e}")

    return results


def search_security_news() -> str:
    from datetime import date
    year = date.today().year
    return web_search(f"cybersecurity news latest vulnerabilities {year}", max_results=5)
