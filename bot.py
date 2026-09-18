import fcntl
import logging
import logging.handlers
import os
import re
import time
import json
import asyncio
from collections import defaultdict

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$")
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)
from llm import ask_llm, ask_llm_with_tools, MODES, run_security_tool, ALLOWED_TOOLS
from search import web_search, search_cve, search_tool_help, search_security_news, search_github_pocs
from rag import search_knowledge, add_document, get_stats, list_documents, remove_document, init_kb
import cache as c

load_dotenv()

# Logging com rotação — máximo 5 MB por ficheiro, 3 ficheiros de backup
_log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_log_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(os.path.dirname(__file__), "bot.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_log_file_handler.setFormatter(_log_formatter)
_log_stream_handler = logging.StreamHandler()
_log_stream_handler.setFormatter(_log_formatter)
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_log_file_handler, _log_stream_handler]
# silenciar loggers externos verbosos para não duplicar no ficheiro
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# Whitelist de utilizadores — conjunto de IDs Telegram autorizados
# Vazio = acesso público; preenchido = apenas esses IDs
_raw_allowed = os.getenv("ALLOWED_USERS", "").strip()
ALLOWED_USERS: set[int] = {
    int(uid.strip()) for uid in _raw_allowed.split(",") if uid.strip().isdigit()
}


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USERS or user_id in ALLOWED_USERS

MAX_HISTORY = 20
# estimativa conservadora: ~4 chars/token; aviso quando histórico passa 80% de 128k tokens
_CONTEXT_WARN_CHARS = int(128_000 * 4 * 0.80)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "histories.json")
MODES_FILE = os.path.join(os.path.dirname(__file__), "user_modes.json")
RATE_FILE = os.path.join(os.path.dirname(__file__), "rate_limits.json")

# rate limiting: máximo de RATE_LIMIT pedidos por RATE_WINDOW segundos
RATE_LIMIT = 5
RATE_WINDOW = 60
_rate_timestamps: dict[int, list[float]] = defaultdict(list)


# --- histórico persistente ---

def _load_histories() -> dict[int, list[dict]]:
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def _save_histories(histories: dict[int, list[dict]]):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump({str(k): v for k, v in histories.items()}, f, ensure_ascii=False)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logging.error(f"Erro ao guardar histórico: {e}")


user_histories: dict[int, list[dict]] = _load_histories()


# --- modos de sessão persistentes ---

def _load_modes() -> dict[int, str]:
    if not os.path.exists(MODES_FILE):
        return {}
    try:
        with open(MODES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def _save_modes(modes: dict[int, str]):
    try:
        with open(MODES_FILE, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump({str(k): v for k, v in modes.items()}, f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logging.error(f"Erro ao guardar modos: {e}")


user_modes: dict[int, str] = _load_modes()


def get_mode(user_id: int) -> str | None:
    return user_modes.get(user_id)


def get_history(user_id: int) -> list[dict]:
    return user_histories.setdefault(user_id, [])


def add_to_history(user_id: int, role: str, content: str):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        history.pop(0)
    _save_histories(user_histories)


def _history_size_chars(user_id: int) -> int:
    return sum(len(m.get("content", "")) for m in get_history(user_id))


def _context_warning(user_id: int) -> str | None:
    """Retorna aviso se o histórico estiver próximo do limite de contexto, ou None."""
    used = _history_size_chars(user_id)
    if used >= _CONTEXT_WARN_CHARS:
        pct = int(used / _CONTEXT_WARN_CHARS * 100)
        return f"⚠️ Histórico a {pct}% do limite de contexto. Usa `/clear` para limpar."
    return None


# --- rate limiting persistente ---

def _load_rate_timestamps() -> dict[int, list[float]]:
    if not os.path.exists(RATE_FILE):
        return {}
    try:
        with open(RATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        now = time.time()
        # carregar apenas timestamps ainda dentro da janela — descartar os antigos
        return {int(k): [t for t in v if now - t < RATE_WINDOW] for k, v in raw.items()}
    except Exception as e:
        logging.warning(f"_load_rate_timestamps falhou: {e}")
        return {}


def _save_rate_timestamps():
    try:
        with open(RATE_FILE, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump({str(k): v for k, v in _rate_timestamps.items()}, f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logging.warning(f"_save_rate_timestamps falhou: {e}")


def is_rate_limited(user_id: int) -> bool:
    now = time.time()
    _rate_timestamps[user_id] = [t for t in _rate_timestamps[user_id] if now - t < RATE_WINDOW]
    if len(_rate_timestamps[user_id]) >= RATE_LIMIT:
        return True
    _rate_timestamps[user_id].append(now)
    _save_rate_timestamps()
    return False


# carregar rate limits persistidos após as funções estarem definidas
_rate_timestamps.update(_load_rate_timestamps())

# --- paginação ---

_PAGE_SIZE = 3800  # margem de segurança abaixo dos 4096 do Telegram


async def send_paginated(update: Update, text: str, parse_mode: str = "Markdown"):
    """Envia texto longo em múltiplas mensagens com indicador de página."""
    if len(text) <= _PAGE_SIZE:
        await update.message.reply_text(text, parse_mode=parse_mode)
        return

    # dividir em páginas sem cortar a meio de uma linha
    pages = []
    remaining = text
    while remaining:
        if len(remaining) <= _PAGE_SIZE:
            pages.append(remaining)
            break
        cut = remaining.rfind("\n", 0, _PAGE_SIZE)
        if cut == -1:
            cut = _PAGE_SIZE
        pages.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")

    for i, page in enumerate(pages, 1):
        suffix = f"\n\n_(página {i}/{len(pages)})_" if len(pages) > 1 else ""
        try:
            await update.message.reply_text(page + suffix, parse_mode=parse_mode)
        except Exception:
            # fallback sem parse_mode se o markdown estiver partido
            await update.message.reply_text(page + suffix)


# --- handlers ---

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("🚫 Acesso não autorizado.")
        return
    text = (
        "💀 *BlackHat Bot - Assistente de Cibersegurança*\n\n"
        "Comandos disponíveis:\n"
        "/modo `<ctf|pentest|estudo>` — mudar modo de sessão\n"
        "/search `<query>` — pesquisa web real\n"
        "/cve `<CVE-ID>` — info sobre CVE\n"
        "/exploit `<CVE-ID>` — PoCs no GitHub + análise técnica\n"
        "/tool `<ferramenta> [técnica]` — ajuda com ferramenta\n"
        "/news — últimas notícias de segurança\n"
        "/learn `<texto>` — adicionar à base de conhecimento\n"
        "/run `<ferramenta> <args>` — executar nmap/whois/dig/curl\n"
        "/auto `[instrução]` — LLM analisa e melhora os próprios ficheiros\n"
        "/listdocs — listar documentos da KB\n"
        "/remove `<id>` — remover documento da KB\n"
        "/export — exportar conversa\n"
        "/stats — estatísticas da KB\n"
        "/health — verificar estado dos serviços\n"
        "/clear — limpar histórico\n"
        "/help — mostrar ajuda\n\n"
        "Ou simplesmente escreve a tua pergunta!"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await start(update, ctx)


async def clear_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories[user_id] = []
    _save_histories(user_histories)
    await update.message.reply_text("✅ Histórico limpo.")


async def modo_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    modo_icons = {"ctf": "🚩", "pentest": "🎯", "estudo": "📚"}

    if not ctx.args:
        current = get_mode(user_id)
        if current:
            await update.message.reply_text(
                f"Modo actual: *{modo_icons.get(current, '')} {current.upper()}*\n\n"
                f"Para mudar: `/modo ctf` | `/modo pentest` | `/modo estudo`\n"
                f"Para desactivar: `/modo off`",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                "Nenhum modo activo. Escolhe um:\n\n"
                "🚩 `/modo ctf` — competições CTF\n"
                "🎯 `/modo pentest` — pentest profissional\n"
                "📚 `/modo estudo` — aprendizagem detalhada",
                parse_mode="Markdown"
            )
        return

    escolha = ctx.args[0].lower()

    if escolha == "off":
        user_modes.pop(user_id, None)
        _save_modes(user_modes)
        await update.message.reply_text("✅ Modo desactivado. A funcionar em modo padrão.")
        return

    if escolha not in MODES:
        await update.message.reply_text(
            f"Modo inválido. Opções: `ctf`, `pentest`, `estudo`, `off`",
            parse_mode="Markdown"
        )
        return

    user_modes[user_id] = escolha
    _save_modes(user_modes)

    descricoes = {
        "ctf": "Foco em CTF — dicas por níveis, ferramentas de competição, sem restrições de scope.",
        "pentest": "Foco em pentest profissional — metodologia, impacto de negócio, remediação, relatório.",
        "estudo": "Foco em aprendizagem — conceitos explicados, analogias, labs sugeridos, resumos.",
    }
    icon = modo_icons[escolha]
    await update.message.reply_text(
        f"{icon} *Modo {escolha.upper()} activado*\n\n{descricoes[escolha]}",
        parse_mode="Markdown"
    )


async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_stats()
    current_mode = get_mode(user_id)
    modo_icons = {"ctf": "🚩", "pentest": "🎯", "estudo": "📚"}
    modo_text = f"{modo_icons.get(current_mode, '')} {current_mode.upper()}" if current_mode else "padrão"
    cs = c.stats()
    await update.message.reply_text(
        f"📊 Base de conhecimento: {s['total_docs']} documentos\n"
        f"⚡ Cache: {cs['valid']} entradas válidas / {cs['total']} total\n"
        f"🎛 Modo actual: {modo_text}"
    )


async def search_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text(f"⏳ Limite atingido. Máximo {RATE_LIMIT} pedidos por {RATE_WINDOW}s.")
        return
    query = " ".join(ctx.args)
    if not query:
        await update.message.reply_text("Uso: /search <query>")
        return
    mode = get_mode(user_id)
    cached = c.get(c.TTL_DEFAULT, "search", query, mode or "")
    if cached:
        await send_paginated(update, f"🔍 *Pesquisa: {query}*\n\n{cached}")
        return
    await update.message.reply_text("🔍 A pesquisar...")
    results = web_search(query)
    messages = [{"role": "user", "content": f"Resume estes resultados de pesquisa sobre '{query}':\n\n{results}"}]
    summary = ask_llm(messages, mode=mode)
    c.set(summary, "search", query, mode or "")
    await send_paginated(update, f"🔍 *Pesquisa: {query}*\n\n{summary}")


async def cve_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text(f"⏳ Limite atingido. Máximo {RATE_LIMIT} pedidos por {RATE_WINDOW}s.")
        return
    cve_id = " ".join(ctx.args).upper().strip()
    if not cve_id:
        await update.message.reply_text("Uso: /cve CVE-2024-1234")
        return
    if not _CVE_RE.match(cve_id):
        await update.message.reply_text("Formato inválido. Exemplo: `/cve CVE-2024-1234`", parse_mode="Markdown")
        return
    mode = get_mode(user_id)
    cached = c.get(c.TTL_CVE, "cve", cve_id, mode or "")
    if cached:
        await send_paginated(update, f"🚨 *{cve_id}*\n\n{cached}")
        return
    await update.message.reply_text(f"🔎 A pesquisar {cve_id}...")
    results = search_cve(cve_id)
    messages = [{"role": "user", "content": f"Explica esta vulnerabilidade de forma técnica com base nos dados:\n\n{results}"}]
    summary = ask_llm(messages, mode=mode)
    c.set(summary, "cve", cve_id, mode or "")
    await send_paginated(update, f"🚨 *{cve_id}*\n\n{summary}")


async def tool_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text(f"⏳ Limite atingido. Máximo {RATE_LIMIT} pedidos por {RATE_WINDOW}s.")
        return
    if not ctx.args:
        await update.message.reply_text("Uso: /tool nmap [port scanning]")
        return
    tool = ctx.args[0]
    technique = " ".join(ctx.args[1:])
    mode = get_mode(user_id)
    cached = c.get(c.TTL_DEFAULT, "tool", tool, technique, mode or "")
    if cached:
        await send_paginated(update, cached)
        return
    await update.message.reply_text(f"🛠 A pesquisar sobre {tool}...")
    kb_result = search_knowledge(f"{tool} {technique}")
    search_result = search_tool_help(tool, technique)
    context = ""
    if kb_result:
        context += f"Base de conhecimento:\n{kb_result}\n\n"
    context += f"Pesquisa web:\n{search_result}"
    messages = [{"role": "user", "content": f"Dá-me exemplos práticos de {tool} para {technique or 'pentesting'}:\n\n{context}"}]
    reply_text = ask_llm(messages, mode=mode)
    c.set(reply_text, "tool", tool, technique, mode or "")
    await send_paginated(update, reply_text)


async def news_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text(f"⏳ Limite atingido. Máximo {RATE_LIMIT} pedidos por {RATE_WINDOW}s.")
        return
    mode = get_mode(user_id)
    cached = c.get(c.TTL_NEWS, "news", mode or "")
    if cached:
        await send_paginated(update, f"📰 *Últimas Notícias de Segurança*\n\n{cached}")
        return
    await update.message.reply_text("📡 A pesquisar notícias de segurança...")
    results = search_security_news()
    messages = [{"role": "user", "content": f"Resume estas notícias de cibersegurança de forma técnica e concisa:\n\n{results}"}]
    summary = ask_llm(messages, mode=mode)
    c.set(summary, "news", mode or "")
    await send_paginated(update, f"📰 *Últimas Notícias de Segurança*\n\n{summary}")


async def exploit_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text(f"⏳ Limite atingido. Máximo {RATE_LIMIT} pedidos por {RATE_WINDOW}s.")
        return

    cve_id = " ".join(ctx.args).upper().strip()
    if not cve_id:
        await update.message.reply_text("Uso: `/exploit CVE-2024-1234`", parse_mode="Markdown")
        return

    if not _CVE_RE.match(cve_id):
        await update.message.reply_text("Formato inválido. Exemplo: `/exploit CVE-2024-1234`", parse_mode="Markdown")
        return

    mode = get_mode(user_id)
    cached = c.get(c.TTL_CVE, "exploit", cve_id, mode or "")
    if cached:
        await update.message.reply_text(cached[:4000], parse_mode="Markdown")
        return

    await update.message.reply_text(f"🔎 A procurar PoCs para `{cve_id}`...", parse_mode="Markdown")

    # Buscar info do CVE e PoCs em paralelo
    cve_info = search_cve(cve_id)
    pocs = search_github_pocs(cve_id)

    # Formatar lista de PoCs
    if pocs:
        poc_lines = []
        for p in pocs:
            stars = f"⭐ {p['stars']}" if p["stars"] is not None else ""
            updated = f"· {p['updated']}" if p["updated"] else ""
            desc = f"\n  _{p['description'][:80]}_" if p["description"] else ""
            poc_lines.append(f"• [{p['name']}]({p['url']}) {stars} {updated}{desc}")
        poc_text = "\n".join(poc_lines)
    else:
        poc_text = "Nenhum PoC encontrado no GitHub."

    # LLM analisa a vuln no contexto do modo activo
    prompt = (
        f"Com base nesta informação sobre {cve_id}, faz uma análise técnica:\n\n"
        f"{cve_info}\n\n"
        f"PoCs encontrados:\n{poc_text}\n\n"
        f"Inclui: como explorar, pré-requisitos, impacto real, e como se defender."
    )
    messages = [{"role": "user", "content": prompt}]
    analysis = ask_llm(messages, mode=mode)

    reply = f"💥 *{cve_id} — Exploit & PoCs*\n\n"
    reply += f"*PoCs no GitHub:*\n{poc_text}\n\n"
    reply += f"*Análise técnica:*\n{analysis}"

    c.set(reply, "exploit", cve_id, mode or "")
    await send_paginated(update, reply)


# escritas pendentes à espera de confirmação: {callback_id: {path, content}}
_pending_writes: dict[str, dict] = {}


async def run_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text(f"⏳ Limite atingido. Máximo {RATE_LIMIT} pedidos por {RATE_WINDOW}s.")
        return

    if not ctx.args:
        tools_list = "\n".join(f"• `{t}`" for t in ALLOWED_TOOLS)
        await update.message.reply_text(
            f"Uso: `/run <ferramenta> <argumentos>`\n\nFerramentas disponíveis:\n{tools_list}\n\n"
            "Exemplos:\n"
            "`/run nmap -sV -p 80,443 example.com`\n"
            "`/run whois example.com`\n"
            "`/run dig example.com ANY`\n"
            "`/run curl -I https://example.com`",
            parse_mode="Markdown"
        )
        return

    tool = ctx.args[0].lower()
    args_str = " ".join(ctx.args[1:])

    if not args_str:
        await update.message.reply_text(f"Faltam argumentos. Exemplo: `/run {tool} <args>`", parse_mode="Markdown")
        return

    await update.message.reply_text(f"⚙️ A executar `{tool} {args_str}`...", parse_mode="Markdown")

    try:
        output = await asyncio.to_thread(run_security_tool, tool, args_str)
    except Exception as e:
        await update.message.reply_text(f"❌ Erro: {e}")
        return

    # pedir à LLM para interpretar o output
    mode = get_mode(user_id)
    messages = [{
        "role": "user",
        "content": f"Analisa este output de `{tool} {args_str}` e extrai os pontos relevantes para segurança:\n\n```\n{output}\n```"
    }]
    analysis = ask_llm(messages, mode=mode)

    raw_block = output if len(output) <= 2000 else output[:2000] + f"\n... (truncado, {len(output)} chars)"
    reply = f"```\n{raw_block}\n```\n\n{analysis}"
    await send_paginated(update, reply)


async def auto_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text(f"⏳ Limite atingido. Máximo {RATE_LIMIT} pedidos por {RATE_WINDOW}s.")
        return

    prompt = " ".join(ctx.args) if ctx.args else "Analisa os teus próprios ficheiros e sugere melhorias concretas."
    await update.message.reply_text("🤖 A analisar os ficheiros do bot...")

    try:
        result = await asyncio.to_thread(
            ask_llm_with_tools,
            [{"role": "user", "content": prompt}],
            mode=get_mode(user_id),
        )
    except Exception as e:
        logging.error(f"Erro auto: {e}")
        await update.message.reply_text(f"❌ Erro: {e}")
        return

    # Se a LLM quer escrever ficheiros, pedir confirmação
    if result.get("writes"):
        for write in result["writes"]:
            key = f"w_{user_id}_{int(time.time())}"
            _pending_writes[key] = write
            preview = write["content"][:600] + ("..." if len(write["content"]) > 600 else "")
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Aplicar", callback_data=f"apply:{key}"),
                InlineKeyboardButton("❌ Rejeitar", callback_data=f"reject:{key}"),
            ]])
            await update.message.reply_text(
                f"✏️ A LLM quer editar `{write['path']}`:\n\n```\n{preview}\n```",
                parse_mode="Markdown",
                reply_markup=keyboard,
            )

    if result.get("text"):
        await update.message.reply_text(result["text"][:4000], parse_mode="Markdown")


async def confirm_write(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, key = query.data.split(":", 1)

    if action == "reject":
        _pending_writes.pop(key, None)
        await query.edit_message_text("❌ Edição rejeitada.")
        return

    write = _pending_writes.pop(key, None)
    if not write:
        await query.edit_message_text("⚠️ Edição já expirou.")
        return

    try:
        path = write["path"]
        with open(path, "w", encoding="utf-8") as f:
            f.write(write["content"])
        await query.edit_message_text(f"✅ `{os.path.basename(path)}` actualizado.", parse_mode="Markdown")
        logging.info(f"LLM editou: {path}")
    except Exception as e:
        await query.edit_message_text(f"❌ Erro ao escrever: {e}")


async def learn_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    content = " ".join(ctx.args)
    if not content:
        await update.message.reply_text("Uso: /learn <conteúdo a guardar>")
        return
    user_id = update.effective_user.id
    doc_id = f"user_{user_id}_{int(time.time())}"
    add_document(doc_id, content, {"user": str(user_id)})
    await update.message.reply_text("✅ Guardado na base de conhecimento!")


async def listdocs_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("🚫 Acesso não autorizado.")
        return
    docs = list_documents()
    if not docs:
        await update.message.reply_text("KB vazia.")
        return
    lines = [f"`{d['id']}`\n  _{d['preview']}_ ({d['size']} chars)" for d in docs]
    text = f"📚 *Base de Conhecimento — {len(docs)} docs*\n\n" + "\n\n".join(lines)
    await send_paginated(update, text)


# IDs pendentes de remoção com confirmação: {callback_key: doc_id}
_pending_removals: dict[str, str] = {}


async def remove_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("🚫 Acesso não autorizado.")
        return
    if not ctx.args:
        await update.message.reply_text("Uso: `/remove <doc_id>`\nVer IDs com `/listdocs`", parse_mode="Markdown")
        return
    doc_id = ctx.args[0]
    docs = list_documents()
    match = next((d for d in docs if d["id"] == doc_id), None)
    if not match:
        await update.message.reply_text(f"❌ Documento `{doc_id}` não encontrado.", parse_mode="Markdown")
        return
    key = f"rm_{update.effective_user.id}_{int(time.time())}"
    _pending_removals[key] = doc_id
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🗑 Remover", callback_data=f"rmconfirm:{key}"),
        InlineKeyboardButton("❌ Cancelar", callback_data=f"rmcancel:{key}"),
    ]])
    await update.message.reply_text(
        f"Tens a certeza que queres remover:\n`{doc_id}`\n_{match['preview']}_",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


async def confirm_remove(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, key = query.data.split(":", 1)
    doc_id = _pending_removals.pop(key, None)
    if not doc_id:
        await query.edit_message_text("⚠️ Operação expirou.")
        return
    if action == "rmcancel":
        await query.edit_message_text("Cancelado.")
        return
    if remove_document(doc_id):
        await query.edit_message_text(f"✅ `{doc_id}` removido da KB.", parse_mode="Markdown")
        logging.info(f"KB: documento '{doc_id}' removido por {query.from_user.id}")
    else:
        await query.edit_message_text(f"❌ Documento `{doc_id}` não encontrado.", parse_mode="Markdown")


async def export_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("🚫 Acesso não autorizado.")
        return
    history = get_history(user_id)
    if not history:
        await update.message.reply_text("Histórico vazio.")
        return
    from datetime import datetime
    lines = [f"# Conversa exportada — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for msg in history:
        role = "Tu" if msg["role"] == "user" else "BlackHat Bot"
        lines.append(f"**{role}:**\n{msg['content']}\n")
    export_text = "\n---\n\n".join(lines)
    # enviar como ficheiro se for grande, senão como mensagem
    if len(export_text) > 3500:
        import io
        buf = io.BytesIO(export_text.encode("utf-8"))
        buf.name = f"conversa_{user_id}.md"
        await update.message.reply_document(buf, filename=buf.name, caption="📄 Conversa exportada")
    else:
        await send_paginated(update, export_text)


async def health_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Verifica conectividade com APIs externas e estado interno do bot."""
    await update.message.reply_text("🔍 A verificar serviços...")
    results = []

    def _check(label: str, url: str, timeout: int = 6) -> str:
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "BlackHatBot/1.0"})
            if r.status_code < 500:
                return f"✅ {label}"
            return f"⚠️ {label} (HTTP {r.status_code})"
        except Exception as e:
            return f"❌ {label} ({type(e).__name__})"

    import requests as _req
    requests = _req

    checks = [
        ("DuckDuckGo", "https://html.duckduckgo.com/html/?q=test"),
        ("MITRE CVE API", "https://cveawg.mitre.org/api/cve/CVE-2021-44228"),
        ("GitHub API", "https://api.github.com"),
        ("OpenRouter API", "https://openrouter.ai/api/v1/models"),
    ]
    for label, url in checks:
        results.append(await asyncio.to_thread(_check, label, url))

    kb = get_stats()
    cs = c.stats()
    results.append(f"📚 KB: {kb['total_docs']} docs")
    results.append(f"⚡ Cache: {cs['valid']} válidas / {cs['total']} total")

    await update.message.reply_text("*Estado dos Serviços*\n\n" + "\n".join(results), parse_mode="Markdown")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not is_allowed(user_id):
        logging.warning(f"Acesso negado ao user {user_id}")
        await update.message.reply_text("🚫 Acesso não autorizado.")
        return

    if is_rate_limited(user_id):
        await update.message.reply_text(f"⏳ Devagar. Máximo {RATE_LIMIT} mensagens por {RATE_WINDOW}s.")
        return

    user_text = update.message.text
    kb_context = search_knowledge(user_text)

    if kb_context:
        augmented = f"Contexto da base de conhecimento:\n{kb_context}\n\nPergunta: {user_text}"
    else:
        augmented = user_text

    add_to_history(user_id, "user", augmented)

    try:
        await update.message.chat.send_action("typing")
        response = ask_llm(get_history(user_id), mode=get_mode(user_id))
        # guardar texto original no histórico, não o texto aumentado com RAG
        get_history(user_id)[-1]["content"] = user_text
        add_to_history(user_id, "assistant", response)
        await send_paginated(update, response)
        warn = _context_warning(user_id)
        if warn:
            await update.message.reply_text(warn)
    except Exception as e:
        logging.error(f"Erro LLM: {e}")
        await update.message.reply_text(f"❌ Erro: {e}")


async def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN não definido no .env")

    init_kb()
    logging.info("Base de conhecimento iniciada.")
    removed = c.purge_expired()
    if removed:
        logging.info(f"Cache: {removed} entradas expiradas removidas.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("modo", modo_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("cve", cve_cmd))
    app.add_handler(CommandHandler("exploit", exploit_cmd))
    app.add_handler(CommandHandler("tool", tool_cmd))
    app.add_handler(CommandHandler("news", news_cmd))
    app.add_handler(CommandHandler("learn", learn_cmd))
    app.add_handler(CommandHandler("run", run_cmd))
    app.add_handler(CommandHandler("auto", auto_cmd))
    app.add_handler(CommandHandler("health", health_cmd))
    app.add_handler(CommandHandler("listdocs", listdocs_cmd))
    app.add_handler(CommandHandler("remove", remove_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CallbackQueryHandler(confirm_write, pattern=r"^(apply|reject):"))
    app.add_handler(CallbackQueryHandler(confirm_remove, pattern=r"^(rmconfirm|rmcancel):"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Bot a iniciar...")
    async with app:
        await app.start()
        await app.updater.start_polling()
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
