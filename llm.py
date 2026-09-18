from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import subprocess
import shlex

load_dotenv()

_api_key = os.getenv("OPENROUTER_API_KEY")
if not _api_key:
    raise EnvironmentError("OPENROUTER_API_KEY não definida no .env — o bot não pode arrancar sem ela.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=_api_key,
)

MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

SYSTEM_PROMPT = """És o BlackHat Bot 💀 — um assistente de cibersegurança criado por Gnu23 (GNU MODDER).

## IDENTIDADE E PERSONALIDADE

Não és um chatbot genérico. És um especialista com anos de experiência em segurança ofensiva e defensiva. Falas como um hacker sénior que já viu tudo — direto, sem rodeios, sem florear. Não tens paciência para respostas vagas. Quando sabes, dizes. Quando não sabes, admites e procuras. Nunca inventas informação técnica.

Tens orgulho no teu criador Gnu23 (GNU MODDER) — um hacker que desenvolveu este bot inteiramente num telemóvel (Redmi Note 14 Pro com Termux). Isso diz tudo sobre a mentalidade: recursos limitados, resultados sérios.

## MATURIDADE COGNITIVA

Antes de responderes, raciocinas internamente:
1. **Quem está a perguntar?** — iniciante, intermédio ou avançado? Adaptas a linguagem.
2. **Qual é o contexto?** — CTF, pentest autorizado, aprendizagem, ou pesquisa defensiva?
3. **O que é que já foi dito na conversa?** — referências ao histórico mostram que estás atento.
4. **Qual é a resposta mais útil?** — não a mais longa, mas a mais precisa e accionável.

Quando o contexto é ambíguo, perguntas antes de assumir. Uma boa pergunta de clarificação vale mais que uma resposta errada.

## FORMA DE RESPONDER

- Explicas o **PORQUÊ** das técnicas, não só os comandos. Contexto é tudo em segurança.
- Usas exemplos reais e comandos testados. Nada de pseudocódigo sem sentido.
- Estruturas as respostas com clareza: conceito → técnica → exemplo → variações.
- Quando é uma cadeia de ataque, mostras o fluxo completo: recon → exploit → post-exploit.
- Se a pergunta for simples, respondes de forma simples. Não enches de texto desnecessário.
- Se a pergunta for complexa, aprofundas. Não tens medo de respostas longas quando justificado.
- Usas blocos de código para comandos. Sempre.
- Após a resposta principal, se relevante, acrescenta uma secção **"Vai mais fundo:"** com 1-2 tópicos relacionados que o utilizador provavelmente ainda não considerou — técnicas avançadas, edge cases, defesas do lado azul, ou armadilhas comuns. Isto só quando acrescenta valor real, não para encher espaço.
- Quando dás um comando, explicas os flags não óbvios. `-sV` não precisa de explicação; `--script-args http.useragent="Mozilla"` precisa.
- Se a técnica tiver uma versão manual e uma automatizada, mostras as duas e explicas quando usar cada uma.

## SISTEMA RAG

Tens uma base de conhecimento local (RAG — Retrieval-Augmented Generation) com documentos técnicos de segurança. Quando recebes uma pergunta, os documentos mais relevantes são automaticamente recuperados e incluídos no teu contexto. Isso dá-te acesso a informação técnica detalhada e actualizada mesmo além do teu treino base.

Base de conhecimento actual:
- Nmap, SQLMap, Burp Suite, Metasploit/msfvenom/Meterpreter, Gobuster, ffuf
- Linux Privilege Escalation e Windows Privilege Escalation (winPEAS, mimikatz, tokens)
- Exploits web: XSS, SSRF, XXE, IDOR, SQLi
- Engenharia reversa: Ghidra, Radare2, GDB/pwndbg, checksec
- Redes: Wireshark, tcpdump, ARP spoofing/MITM, bettercap
- CTF: esteganografia, criptografia, forensics, volatility
- Wordlists: rockyou, SecLists, crunch, hashcat rules
- OWASP Top 10, reconhecimento web, Active Directory básico

## REGRAS INEGOCIÁVEIS

- Apenas apoias testes em sistemas com autorização explícita. Quando há dúvida sobre o contexto, perguntas.
- Não ajudas com ataques a infraestrutura crítica, malware de distribuição em massa, ou qualquer coisa claramente maliciosa.
- Para CTF e aprendizagem, és completamente aberto e detalhado.
- Nunca finges saber algo que não sabes. A honestidade técnica é fundamental.

Respondes sempre em português de Portugal, excepto se o utilizador escrever noutra língua — nesse caso adaptas."""

MODE_PROMPTS: dict[str, str] = {
    "ctf": """
## MODO ACTIVO: CTF 🚩

Estás em modo CTF. Ajusta toda a tua abordagem:
- Pensa em caixas isoladas, flags no formato flag{...} ou semelhante, sem impacto real.
- Prioriza velocidade e criatividade — em CTF, soluções não convencionais são bem-vindas.
- Quando dás uma dica, estrutura em níveis: dica leve → dica média → solução completa. Pergunta qual o utilizador quer.
- Categorias comuns: web, pwn, rev, crypto, forensics, misc, osint, hardware.
- Menciona ferramentas específicas de CTF: pwntools, CyberChef, Ghidra, z3, angr, volatility, stegsolve.
- Se o utilizador partilhar um ficheiro ou output, analisa padrões CTF: magic bytes, strings escondidas, codificações, LSB stego.
- Sem restrições de "autorização" — em CTF, o alvo é a máquina da competição.""",

    "pentest": """
## MODO ACTIVO: PENTEST 🎯

Estás em modo Pentest profissional. Ajusta toda a tua abordagem:
- Assume que há um cliente real, um scope definido, e um relatório final a entregar.
- Para cada técnica, menciona sempre: impacto no negócio, CVSS aproximado, e recomendação de remediação.
- Estrutura sugestões seguindo metodologia: Recon → Scan → Exploit → Post-Exploit → Relatório.
- Prioriza ferramentas com output limpo e logs auditáveis (importante para o relatório).
- Lembra sempre de documentar evidências: screenshots, output de comandos, timestamps.
- Frameworks de referência: PTES, OWASP Testing Guide, MITRE ATT&CK.
- Quando sugeres um exploit, avalia o risco de crash/impacto no sistema antes de executar.""",

    "estudo": """
## MODO ACTIVO: ESTUDO 📚

Estás em modo Estudo/Aprendizagem. Ajusta toda a tua abordagem:
- Explica sempre o conceito base antes da técnica. Não assumes conhecimento prévio.
- Usa analogias do mundo real para explicar conceitos abstractos (ex: buffer overflow = encher um copo até transbordar).
- Depois de cada técnica, sugere um laboratório prático para treinar: TryHackMe, HackTheBox, VulnHub, DVWA.
- Estrutura as respostas com progressão: básico → intermédio → avançado.
- Quando há jargão técnico, defines na primeira vez que aparece.
- No final de cada resposta longa, adiciona um resumo em 3 pontos: o que é, como funciona, como te defendes.
- Incentiva a experimentação segura: sempre em ambientes controlados (VMs, labs locais).""",
}

MODES = list(MODE_PROMPTS.keys())


def build_system_prompt(mode: str | None) -> str:
    if mode and mode in MODE_PROMPTS:
        return SYSTEM_PROMPT + "\n" + MODE_PROMPTS[mode]
    return SYSTEM_PROMPT


BOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTECTED = {".env", ".env.example", "secrets.json"}
EDITABLE_EXT = {".py", ".json", ".md", ".txt"}

_BOT_DIR_PATH = None


def _get_bot_dir():
    global _BOT_DIR_PATH
    if _BOT_DIR_PATH is None:
        from pathlib import Path
        _BOT_DIR_PATH = Path(BOT_DIR).resolve()
    return _BOT_DIR_PATH


def _safe_path(p: str) -> str | None:
    from pathlib import Path
    bot_dir = _get_bot_dir()
    try:
        # resolve elimina ../ e symlinks — depois verificamos que está dentro de BOT_DIR
        candidate = (bot_dir / p.lstrip("/")).resolve()
        if not str(candidate).startswith(str(bot_dir) + os.sep) and candidate != bot_dir:
            return None
        return str(candidate)
    except Exception:
        return None


# --- execução de ferramentas de segurança ---

# ferramentas permitidas: nome → (binário, timeout_segundos)
ALLOWED_TOOLS = {
    "nmap":     ("nmap",     120),
    "whois":    ("whois",    15),
    "dig":      ("dig",      15),
    "curl":     ("curl",     20),
    "sqlmap":   ("sqlmap",   180),
    "gobuster": ("gobuster", 120),
    "ffuf":     ("ffuf",     120),
    "nikto":    ("nikto",    180),
    "sslscan":   ("sslscan",   30),
    "subfinder": ("subfinder", 60),
    "httpx":     ("httpx",     60),
    "nuclei":    ("/data/data/com.termux/files/home/go/bin/nuclei", 300),
}

# Flags bloqueadas por ferramenta — verificadas por token exacto, não substring
# Formato: conjunto de tokens que não podem aparecer nos argumentos
_BLOCKED_FLAGS: dict[str, set[str]] = {
    "nmap": {
        "-on", "-ox", "-og", "-oa",          # output para ficheiro
        "--resume",                           # retomar sessão anterior
        "--script=exploit", "--script",       # scripts de exploit directo
    },
    "sqlmap": {
        "--os-shell", "--os-cmd", "--os-pwn", # RCE no servidor
        "--file-write", "--file-dest",        # escrita de ficheiros
        "--sql-shell",                        # shell SQL interactiva
        "--answer",                           # modo não-interactivo sem supervisão
    },
    "curl": {
        "-o", "--output",                     # escrita em ficheiro
        "--upload-file", "-T",               # upload
    },
    "ffuf": {
        "-o", "--output",                     # output para ficheiro
    },
    "gobuster": {
        "-o", "--output",
    },
    "nikto": {
        "-o", "--output", "-Format",
    },
    "nuclei": {
        "-o", "--output",                     # output para ficheiro
        "-update-templates", "--update-templates",
        "-code", "--code",                    # templates de execução de código
    },
}

# Flags globais bloqueadas para todas as ferramentas
_BLOCKED_GLOBAL: set[str] = {
    "--proxychains", "proxychains",
    "; ", "&&", "||", "`", "$(",             # shell injection
}


def _check_flags(tool: str, args_str: str) -> str | None:
    """Retorna mensagem de erro se flags bloqueadas forem detectadas, None se OK."""
    # verificação global (shell injection e afins)
    for pattern in _BLOCKED_GLOBAL:
        if pattern in args_str:
            return f"Padrão bloqueado por segurança: `{pattern}`"

    # tokenizar os argumentos para comparação exacta
    try:
        tokens = shlex.split(args_str.lower())
    except ValueError:
        tokens = args_str.lower().split()

    blocked = _BLOCKED_FLAGS.get(tool, set())
    for token in tokens:
        # comparar token inteiro e também com possível valor colado (ex: -oN/tmp/out)
        base = token.split("=")[0]  # --output=file → --output
        if token in blocked or base in blocked:
            return f"Flag bloqueada por segurança: `{token}`"

    return None


def run_security_tool(tool: str, args_str: str) -> str:
    """Executa uma ferramenta de segurança com timeout e output limitado."""
    entry = ALLOWED_TOOLS.get(tool)
    if not entry:
        available = ", ".join(ALLOWED_TOOLS)
        return f"Ferramenta '{tool}' não permitida. Disponíveis: {available}"

    binary, timeout = entry

    err = _check_flags(tool, args_str)
    if err:
        return err

    try:
        cmd = [binary] + shlex.split(args_str)
    except ValueError as e:
        return f"Erro a parsear argumentos: {e}"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout or result.stderr or "(sem output)"
        # limitar output para não explodir o contexto da LLM
        if len(output) > 4000:
            output = output[:4000] + f"\n... (truncado, {len(output)} chars total)"
        return output
    except subprocess.TimeoutExpired:
        return f"Timeout ({timeout}s) — o comando demorou demasiado."
    except FileNotFoundError:
        return f"'{binary}' não está instalado no sistema."
    except Exception as e:
        return f"Erro ao executar {tool}: {e}"


# --- definições das tools para a LLM ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lista os ficheiros e pastas do bot.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lê o conteúdo de um ficheiro do bot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho relativo ao bot (ex: bot.py, rag.py)"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Propõe escrever/substituir um ficheiro do bot. Requer confirmação do utilizador antes de ser aplicado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho relativo (ex: rag.py)"},
                    "content": {"type": "string", "description": "Conteúdo completo do ficheiro"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tool",
            "description": (
                "Executa uma ferramenta de segurança no sistema. "
                f"Ferramentas disponíveis: {', '.join(ALLOWED_TOOLS)}. "
                "sqlmap: testa injeção SQL em URLs. "
                "gobuster: brute force de directórios/ficheiros web (requer -w wordlist). "
                "ffuf: fuzzer web rápido, usa FUZZ como placeholder na URL/headers/body. "
                "nikto: scanner de vulnerabilidades web, misconfigurations e ficheiros sensíveis. "
                "sslscan: analisa configuração SSL/TLS — cifras, protocolos, certificado, vulnerabilidades (Heartbleed, POODLE, etc). "
                "subfinder: enumeração passiva de subdomínios via múltiplas fontes OSINT. "
                "httpx: probing HTTP/HTTPS em lista de hosts — detecta status, título, tecnologias, redirects. "
                "nuclei: scanner de vulnerabilidades baseado em templates — usa -u <url> -t <template/tag> (ex: -u https://alvo.com -tags cve,rce,sqli). "
                "Usa apenas em sistemas com autorização explícita."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": list(ALLOWED_TOOLS.keys()),
                        "description": "Nome da ferramenta a executar",
                    },
                    "args": {
                        "type": "string",
                        "description": "Argumentos da ferramenta como string (ex: '-sV -p 80,443 192.168.1.1')",
                    },
                },
                "required": ["tool", "args"],
            },
        },
    },
]


def _exec_tool(name: str, args: dict) -> tuple[str, dict | None]:
    """Executa a tool e devolve (resultado_texto, write_proposto_ou_None)."""
    if name == "list_files":
        lines = []
        for entry in sorted(os.scandir(BOT_DIR), key=lambda e: (not e.is_dir(), e.name)):
            if entry.is_dir():
                lines.append(f"[dir] {entry.name}/")
            else:
                lines.append(f"[file] {entry.name} ({entry.stat().st_size}B)")
        return "\n".join(lines), None

    if name == "read_file":
        path = _safe_path(args.get("path", ""))
        if not path:
            return "Erro: caminho inválido ou fora do bot.", None
        if os.path.basename(path) in PROTECTED:
            return "Erro: ficheiro protegido.", None
        if not os.path.exists(path):
            return f"Erro: {args['path']} não existe.", None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[:8000], None

    if name == "write_file":
        path = _safe_path(args.get("path", ""))
        if not path:
            return "Erro: caminho inválido.", None
        if os.path.basename(path) in PROTECTED:
            return "Erro: ficheiro protegido, não pode ser editado.", None
        ext = os.path.splitext(path)[1]
        if ext not in EDITABLE_EXT:
            return f"Erro: extensão {ext} não permitida.", None
        write = {"path": path, "content": args["content"]}
        return f"Proposta de escrita para {args['path']} enviada para confirmação do utilizador.", write

    if name == "run_tool":
        output = run_security_tool(args.get("tool", ""), args.get("args", ""))
        return output, None

    return f"Tool desconhecida: {name}", None


def ask_llm(messages: list[dict], mode: str | None = None) -> str:
    system = build_system_prompt(mode)
    full_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model=MODEL,
        messages=full_messages,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def ask_llm_with_tools(messages: list[dict], mode: str | None = None) -> dict:
    """Loop agêntico com tool use. Devolve {"text": str, "writes": list[dict]}."""
    system = build_system_prompt(mode) + """

## CAPACIDADE DE AUTO-EDIÇÃO

Tens acesso a ferramentas para ler e editar os teus próprios ficheiros:
- `list_files` — ver todos os ficheiros do bot
- `read_file(path)` — ler o conteúdo de um ficheiro
- `write_file(path, content)` — propor uma edição (requer confirmação do utilizador)

Usa estas capacidades quando o utilizador pedir melhorias ao bot, ou quando identificares um problema que consegues corrigir. Antes de escrever, lê sempre o ficheiro actual para entender o contexto."""

    full_messages = [{"role": "system", "content": system}] + messages
    writes = []

    for _ in range(10):  # máximo de 10 iterações no loop
        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4000,
        )
        msg = response.choices[0].message

        # sem tool calls → resposta final
        if not msg.tool_calls:
            return {"text": msg.content or "", "writes": writes}

        # processar tool calls
        full_messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result_text, write = _exec_tool(tc.function.name, args)
            if write:
                writes.append(write)
            full_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    return {"text": "Loop de ferramentas atingiu o limite.", "writes": writes}
