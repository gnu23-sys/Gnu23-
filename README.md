# BlackHat Bot 💀

Assistente de cibersegurança e pentesting para Telegram, desenvolvido por **GNU MODDER** inteiramente num telemóvel (Redmi Note 14 Pro com Termux).

## Funcionalidades

- **Chat inteligente** com contexto de conversa e múltiplos modos (CTF, Pentest, Estudo)
- **RAG** — base de conhecimento local com BM25 + expansão semântica
- **Ferramentas de pentest** executadas directamente via `/run`: nmap, sqlmap, gobuster, ffuf, nikto, nuclei, subfinder, httpx, sslscan, whois, dig, curl
- **Pesquisa web** em tempo real (DuckDuckGo)
- **CVEs** — info técnica + PoCs do GitHub via `/cve` e `/exploit`
- **Auto-edição** — LLM lê e propõe melhorias aos próprios ficheiros via `/auto`
- **Gestão da KB** — `/learn`, `/listdocs`, `/remove`
- **Exportação** de conversas via `/export`
- **Health check** de todos os serviços via `/health`

## Comandos

| Comando | Descrição |
|---|---|
| `/modo <ctf\|pentest\|estudo>` | Mudar modo de sessão |
| `/search <query>` | Pesquisa web real |
| `/cve <CVE-ID>` | Info técnica sobre CVE |
| `/exploit <CVE-ID>` | PoCs GitHub + análise |
| `/tool <ferramenta>` | Ajuda com ferramenta de pentest |
| `/run <ferramenta> <args>` | Executar ferramenta directamente |
| `/news` | Últimas notícias de segurança |
| `/learn <texto>` | Adicionar à base de conhecimento |
| `/listdocs` | Listar documentos da KB |
| `/remove <id>` | Remover documento da KB |
| `/export` | Exportar conversa |
| `/auto [instrução]` | LLM melhora os próprios ficheiros |
| `/stats` | Estatísticas da KB e cache |
| `/health` | Estado dos serviços externos |
| `/clear` | Limpar histórico |

## Base de Conhecimento

48 documentos técnicos incluídos:

**Segurança Ofensiva:** nmap, sqlmap, metasploit, burpsuite, gobuster, ffuf, nuclei, nikto, subfinder, httpx, buffer overflow, heap exploitation, web shells, C2 frameworks, malware development, AV/EDR evasion, firewall evasion

**Web:** OWASP Top 10, XSS avançado, SQL injection, HTTP smuggling, web cache poisoning, subdomain takeover, API hacking, recon automatizado

**Infra:** Linux/Windows privesc, Active Directory, Docker/Kubernetes escape, cloud AWS, IoT hacking, mobile hacking

**OSINT & Recon:** OSINT, social engineering, threat modeling, bug bounty

**IA & Sociedade:** visão geral de IA, impacto social, ética e bias, geopolítica, saúde/ciência, Portugal/Europa, riscos e futuro

## Instalação

### Requisitos

```bash
pip install python-telegram-bot openai python-dotenv requests
```

Para nuclei (opcional):
```bash
go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
nuclei -update-templates
```

### Configuração

Cria um ficheiro `.env` na raiz do projecto:

```env
TELEGRAM_TOKEN="<token do BotFather>"
OPENROUTER_API_KEY="<chave da OpenRouter>"
OPENROUTER_MODEL=openai/gpt-4o-mini
# IDs Telegram autorizados (separados por vírgula) — vazio = público
ALLOWED_USERS=<teu_telegram_id>
```

### Executar

```bash
python3 bot.py
```

## Arquitectura

```
bot.py          — handlers Telegram, rate limiting, paginação
llm.py          — cliente LLM, tool use, execução de ferramentas
rag.py          — BM25 + stop words + expansão semântica
cache.py        — cache em memória com TTL por entrada e file lock
search.py       — DuckDuckGo, MITRE CVE API, GitHub API com retry
knowledge_base.json — base de conhecimento
```

## Segurança

- Whitelist de utilizadores via `ALLOWED_USERS`
- Path traversal prevenido com `pathlib.Path.resolve()`
- Flags bloqueadas por token exacto (não substring) por ferramenta
- Shell injection bloqueado (`;`, `&&`, `||`, backticks)
- File locks em todas as escritas concorrentes
- Validação de CVE com regex `^CVE-\d{4}-\d{4,}$`
- `.env` e dados de utilizador excluídos do repositório

## Créditos

Desenvolvido por **GNU MODDER** — hacker que prova que recursos limitados não limitam resultados.
