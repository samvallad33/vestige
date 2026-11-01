"""Execute source-pinned product hook fragments in an isolated harness client.

Original scripts/settings are copied byte-for-byte. Native HTTP and native CLI
owner-IPC target the *same* stdio child; these compatibility transports are
explicitly logged. No legacy retrieval operation is synthesized. Model bridges
accept only the owner-selected nano callback, never ambient Claude or Ollama.
"""
from __future__ import annotations
import hashlib
import fcntl
import secrets
import socketserver
import urllib.parse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.request

FRAGMENTS=("settings.preflight.fragment.json","settings.rules.fragment.json",
           "settings.sanhedrin.fragment.json","settings.stop-memory.fragment.json")
EVENTS={"SessionStart","UserPromptSubmit","PreToolUse","PostToolUse","Stop"}
MODEL="gpt-5.4-nano"


class HookViolation(RuntimeError):pass


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1",0));return s.getsockname()[1]


def read_audit(repo,path):
    audit=json.loads(Path(path).read_text())
    if audit.get("audit_complete") is not True:raise HookViolation("hook source audit is incomplete")
    pins=audit.get("source_pins") or {}
    required={str(p.relative_to(repo)) for p in (repo/"hooks").iterdir()
              if p.suffix in {".sh",".py"} or p.name in FRAGMENTS}
    if required-set(pins):raise HookViolation("hook audit omits source files: "+repr(sorted(required-set(pins))))
    for name,sha in pins.items():
        p=(repo/name).resolve()
        if not p.is_relative_to(repo):raise HookViolation("hook audit source escapes repository")
        if digest(p)!=sha:raise HookViolation("hook source changed after audit: "+name)
    return audit


def merge_fragments(repo):
    merged={event:[] for event in EVENTS}
    wired=[]
    for name in FRAGMENTS:
        path=repo/"hooks"/name
        for event,groups in json.loads(path.read_text()).get("hooks",{}).items():
            if event not in EVENTS:raise HookViolation("unsupported hook event: "+event)
            for group in groups:
                for hook in group.get("hooks",[]):
                    if hook.get("type")!="command":raise HookViolation("non-command hook needs native client adapter")
                    entry={**hook,"matcher":group.get("matcher"),"fragment":str(path.relative_to(repo))}
                    merged[event].append(entry);wired.append(entry)
    # Fresh owner instruction enables these two generic public hooks in
    # addition to (without editing) the shipped settings fragments.
    for script in ("synthesis-gate.sh", "load-all-memory.sh"):
        entry={"type":"command","command":"$HOME/.claude/hooks/"+script,"timeout":8,
               "matcher":None,"fragment":"owner-all-applicable-hooks-profile", "source":"hooks/"+script}
        merged["UserPromptSubmit"].append(entry);wired.append(entry)
    return merged,wired


class HookController:
    def __init__(self,engine,audit_path=None,model_callback=None):
        self.engine=engine;self.repo=engine.repo;self.root=engine.run_dir/"hooks-runtime"
        self.audit=read_audit(self.repo,audit_path or self.repo/"runs/hook-audit.json")
        self.events,self.wired=merge_fragments(self.repo)
        self.model_callback=model_callback
        if hasattr(model_callback,"bind_engine"):
            model_callback.bind_engine(engine)
        self.dashboard_port=free_port();self.bridge_token=secrets.token_hex(32);self.errors=[];self.server=None;self.thread=None
        self.transcript=self.root/"transcript.jsonl"
        self.log_path=self.root/"hooks.jsonl"
        self.session="trial-"+engine.run_dir.name
        self.native_status={}
        self.cli_server=None;self.cli_lock=None;self._cwd_scope=None
        self.pending_context=[]

    def log(self,data):
        with self.log_path.open("a") as f:
            f.write(json.dumps({"timestamp":time.time(),**data},ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())

    def prepare(self):
        if self.root.exists():raise HookViolation("hook runtime must be fresh")
        home=self.root/"home";self.home=home
        for sub in [".claude/hooks",".vestige/hooks",".config",".local/share",".cache","bin"]:
            (home/sub).mkdir(parents=True,exist_ok=True)
        for sub in ["tmp","pulse","sanhedrin","task-files"]:(self.root/sub).mkdir()
        for path in (self.repo/"hooks").iterdir():
            if path.is_file():
                for dest in [home/".claude/hooks"/path.name,home/".vestige/hooks"/path.name]:
                    shutil.copyfile(path,dest);os.chmod(dest,0o700 if path.suffix in {".sh",".py"} else 0o600)
                    if digest(dest)!=digest(path):raise HookViolation("hook copy differs")
        # Physically outside the checkout: /usr/bin/git AND the original
        # Rust CLI both walk ancestors, so an empty nested directory leaks
        # harness commit/task names even with an isolated HOME.
        scratch_base="/private/tmp" if Path("/private/tmp").is_dir() else "/tmp"
        self._cwd_scope=tempfile.TemporaryDirectory(prefix="vestige-hook-cwd-",dir=scratch_base)
        self.cwd=Path(self._cwd_scope.name).resolve()
        if any((p/".git").exists() for p in (self.cwd,*self.cwd.parents)):
            raise HookViolation("hook working directory has a Git checkout ancestor")
        profile={"hooks":{event:[{"matcher":entry["matcher"],"hooks":[{key:entry[key] for key in ("type","command","timeout")}]} for entry in entries] for event,entries in self.events.items()}}
        self.profile_path=home/".claude/settings.json"
        self.profile_path.write_text(json.dumps(profile,sort_keys=True,indent=2)+"\n")
        self.profile_sha=digest(self.profile_path)
        cli=self.repo/"target/release/vestige"
        if not cli.is_file() or cli.is_symlink():raise HookViolation("repo-built vestige CLI required for native SessionStart hook")
        self.cli_sha=digest(cli)
        self.server=self._server();self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.control_url=f"http://127.0.0.1:{self.server.server_port}"
        self._prepare_cli_transport()
        # The executable is a transport-only shim. It holds no memory or
        # retrieval implementation; all tools/call requests reach this child.
        proxy="""#!PYTHON
import json,os,sys,urllib.request
base=os.environ['VESTIGE_HOOK_CONTROL']
for line in sys.stdin:
 try:
  value=json.loads(line)
  req=urllib.request.Request(base+'/mcp',data=json.dumps(value).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['VESTIGE_HOOK_CONTROL_TOKEN']})
  with urllib.request.urlopen(req,timeout=120) as response:out=json.load(response)
  if value.get('id') is not None:print(json.dumps(out),flush=True)
 except Exception as exc:
  print(str(exc),file=sys.stderr);sys.exit(1)
""".replace("#!PYTHON","#!"+sys.executable)
        (home/"bin/vestige-mcp").write_text(proxy);os.chmod(home/"bin/vestige-mcp",0o700)
        claude="""#!PYTHON
import json,os,sys,urllib.request
args=sys.argv[1:]
if '-p' not in args:raise SystemExit('unsupported Claude transport shape')
prompt=args[args.index('-p')+1]
allowed=args[args.index('--allowed-tools')+1].split(',') if '--allowed-tools' in args else []
requested=args[args.index('--model')+1] if '--model' in args else ''
req=urllib.request.Request(os.environ['VESTIGE_HOOK_CONTROL']+'/lateral',data=json.dumps({'prompt':prompt,'allowed_tools':allowed,'requested_model':requested}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['VESTIGE_HOOK_CONTROL_TOKEN']})
with urllib.request.urlopen(req,timeout=120) as response:value=json.load(response)
print(value['output'])
""".replace("#!PYTHON","#!"+sys.executable)
        (home/"bin/claude").write_text(claude);os.chmod(home/"bin/claude",0o700)
        # Preserve the exact fragment command while pinning CLI and store.
        wrapper='#!/bin/sh\nexec "'+str(cli)+'" --data-dir "$VESTIGE_HOOK_CLI_TRANSPORT_DIR" "$@"\n'
        (home/"bin/vestige").write_text(wrapper);os.chmod(home/"bin/vestige",0o700)
        for name in ["python3"]:
            (home/"bin"/name).symlink_to(sys.executable)
        self.env={"HOME":str(home),"USER":"benchmark-worker","PATH":str(home/"bin")+":/usr/bin:/bin:/usr/sbin:/sbin",
            "XDG_CONFIG_HOME":str(home/".config"),"XDG_DATA_HOME":str(home/".local/share"),"XDG_CACHE_HOME":str(home/".cache"),
            "TMPDIR":str(self.root/"tmp"),"GIT_CEILING_DIRECTORIES":str(self.cwd),"GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":"/dev/null","OPERATOR_HOME":str(self.engine.run_dir/"operator"),
            "OPERATOR_MODE":self.engine.identity.get("operator_policy",{}).get("effective_mode","enforce"),
            "VESTIGE_DATA_DIR":str(self.engine.run_dir/"strata"),"VESTIGE_MCP_BIN":str(home/"bin/vestige-mcp"),
            "VESTIGE_HOOK_CONTROL":self.control_url,"VESTIGE_HOOK_CONTROL_TOKEN":self.bridge_token,"VESTIGE_HOOK_CLI_TRANSPORT_DIR":str(self.cli_data),
            "VESTIGE_DASHBOARD_PORT":str(self.server.server_port),
            "VESTIGE_BASE_URL":self.control_url,
            "VESTIGE_PULSE_STATE_DIR":str(self.root/"pulse"),"VESTIGE_SANHEDRIN_STATE_DIR":str(self.root/"sanhedrin"),
            "VESTIGE_SANHEDRIN_ENV":str(self.root/"no-ambient-config.env"),"VESTIGE_SANHEDRIN_ENABLED":"1",
            "VESTIGE_SANHEDRIN_MODEL":MODEL,"VESTIGE_SANHEDRIN_API_KEY":self.bridge_token,"VESTIGE_SANHEDRIN_ENDPOINT":self.control_url+"/v1/chat/completions",
            "VESTIGE_SANHEDRIN_BACKEND":"openai","VESTIGE_SANHEDRIN_PYTHON":sys.executable,
            "VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER":"0","VESTIGE_SANHEDRIN_ALLOW_LOOSE_LEDGER":"0",
            "VESTIGE_MEMORY_DIR":str(home/".vestige/unused-unwired-memory")}
        self.log({"kind":"prepared","wired":self.wired,"source_pins":self.audit["source_pins"],
            "cli_sha256":self.cli_sha,"profile_sha256":self.profile_sha,"adapter_sha256":digest(Path(__file__)),"model":MODEL,"legacy_model_labels_preserved_in_source":True,
            "memory_transports":["stdio-jsonrpc","HTTP-envelope-to-child-stdio","native-CLI-owner-IPC-envelope-to-child-stdio"],
            "paths":{"home":str(home),"cwd":str(self.cwd)},
            "cwd_git_ancestor_check":{"checked":[str(p) for p in (self.cwd,*self.cwd.parents)],"git_ancestors":[]}})
        return {**{k:v for k,v in self.env.items() if k in {"HOME","PATH","XDG_DATA_HOME","XDG_CONFIG_HOME","XDG_CACHE_HOME","TMPDIR","OPERATOR_HOME","OPERATOR_MODE"}},
            "VESTIGE_DASHBOARD_ENABLED":"true","VESTIGE_DASHBOARD_PORT":str(self.dashboard_port)}

    def _server(self):
        controller=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_GET(self):
                try:
                    parts=urllib.parse.urlsplit(self.path)
                    if parts.path=="/api/health":
                        result=controller.native_call("memory_status",{"view":"health"})
                        status=200
                    elif parts.path=="/api/memories":
                        query=urllib.parse.parse_qs(parts.query)
                        if set(query)-{"tag","limit"} or "tag" not in query:raise HookViolation("unsupported legacy memory request shape")
                        limit=min(200,max(1,int(query.get("limit",["50"])[0])))
                        recalled=controller.native_call("recall",{"handle":query["tag"][0],"limit":limit,"detail_level":"full"})
                        nodes=[]
                        for node in recalled.get("nodes",[]):
                            full=controller.native_call("memory",{"action":"get","id":node["id"]})
                            if full.get("found") is True:nodes.append(full["node"])
                        result={"memories":nodes};status=200
                    elif parts.path=="/api/changelog":
                        status,result=controller.native_unavailable(self.path,None)
                    else:raise HookViolation("unsupported hook GET endpoint")
                    body=json.dumps(result).encode();self.send_response(status);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(body)
                except BaseException as exc:
                    controller.errors.append(str(exc));controller.log({"kind":"bridge_error","path":self.path,"error":str(exc)})
                    self.send_response(500);self.end_headers()
            def do_POST(self):
                if self.path in {"/mcp","/lateral","/v1/chat/completions"} and not secrets.compare_digest(self.headers.get("Authorization", ""),"Bearer "+controller.bridge_token):
                    self.send_response(403);self.end_headers();return
                try:
                    value=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
                    if self.path=="/api/deep_reference":
                        status,result=controller.native_unavailable(self.path,value)
                        self.send_response(status);self.end_headers();self.wfile.write(json.dumps(result).encode());return
                    if self.path=="/mcp":
                        result=controller.forward_mcp(value)
                    elif self.path in {"/lateral","/v1/chat/completions"}:
                        if controller.model_callback is None:raise HookViolation("hook nano model callback not configured; no ambient model permitted")
                        if self.path=="/lateral":
                            payload={"model":MODEL,"messages":[{"role":"user","content":value["prompt"]}],
                                "max_tokens":1024,"allowed_tools":value["allowed_tools"],"hook":"preflight-swarm",
                                "requested_model":value.get("requested_model"),"tool_call_budget":3}
                        else:
                            if value.get("model")!=MODEL:raise HookViolation("Sanhedrin requested non-nano model")
                            payload={**value,"hook":"sanhedrin"}
                        controller.log({"kind":"model_bridge_request","payload":payload,"effective_model":MODEL})
                        result=controller.model_callback(payload)
                        if self.path=="/lateral":result={"output":result["choices"][0]["message"].get("content") or ""}
                    else:raise HookViolation("unsupported hook transport endpoint")
                    body=json.dumps(result).encode();self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(body)
                except BaseException as exc:
                    controller.errors.append(str(exc));controller.log({"kind":"bridge_error","path":self.path,"error":str(exc)})
                    self.send_response(500);self.end_headers();self.wfile.write(json.dumps({"error":str(exc)}).encode())
        return ThreadingHTTPServer(("127.0.0.1",0),Handler)

    def _prepare_cli_transport(self):
        """A native owner-IPC envelope proxy, never a second memory store.

        The unchanged CLI sees a held owner lock and an attach endpoint. Its
        original initialize/tools/call JSON-RPC is forwarded to the existing
        child stdio transport; no log segments or memory bytes live here.
        """
        self.cli_data=self.root/"cli-transport";self.cli_data.mkdir(mode=0o700)
        (self.cli_data/"log").mkdir()
        self.cli_lock=(self.cli_data/".serve.lock").open("w")
        os.chmod(self.cli_data/".serve.lock",0o600)
        fcntl.flock(self.cli_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        token=secrets.token_hex(32);controller=self
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                if self.rfile.readline(257).decode().strip()!="vestige-attach/1 "+token:return
                client=controller.engine.client
                if not client or not client._proc or client._proc.poll() is not None:return
                self.wfile.write(f"vestige-attached/1 {client._proc.pid}\n".encode());self.wfile.flush()
                for line in self.rfile:
                    value=json.loads(line);result=controller.forward_mcp(value)
                    if value.get("id") is not None:self.wfile.write((json.dumps(result)+"\n").encode());self.wfile.flush()
        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address=True
            daemon_threads=True
        self.cli_server=Server(("127.0.0.1",0),Handler)
        threading.Thread(target=self.cli_server.serve_forever,daemon=True).start()
        endpoint=self.cli_data/".serve.endpoint"
        endpoint.write_text(f"{self.cli_server.server_address[1]} {token} {os.getpid()}\n");os.chmod(endpoint,0o600)

    def native_call(self,name,arguments):
        # This low-level path avoids recursive lifecycle hooks while preserving
        # every native request/result in the same child's transport audit.
        with self.engine._call_lock:return self.engine._call(name,arguments)

    def native_unavailable(self,path,body):
        url=f"http://127.0.0.1:{self.dashboard_port}"+path
        request=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(request,timeout=2) as response:
                status,data=response.status,response.read().decode()
        except urllib.error.HTTPError as exc:status,data=exc.code,exc.read().decode()
        if status<400:
            raise HookViolation("legacy endpoint unexpectedly became available; requires new stdio adapter audit")
        self.log({"kind":"native_unavailable_endpoint","path":path,"status":status,"response":data})
        try:parsed=json.loads(data)
        except json.JSONDecodeError:parsed={"native_body":data}
        return status,parsed

    def forward_mcp(self,value):
        client=self.engine.client
        if not client or not client._proc or client._proc.poll() is not None:raise HookViolation("hook engine process is not alive")
        with self.engine._call_lock:
            method=value["method"];params=value.get("params") or {}
            self.log({"kind":"hook_mcp_request","request":value})
            if value.get("id") is None:
                client.notify(method,params);return {}
            result=client.request(method,params)
            self.log({"kind":"hook_mcp_result","result":result})
            return {"jsonrpc":"2.0","id":value["id"],"result":result}

    def probe_native(self):
        # Observe the real child endpoints, without inventing replacement
        # behavior for withheld legacy APIs. Results are telemetry, not scores.
        for path,body in [("/api/health",None),("/api/changelog",None),
                          ("/api/deep_reference",{"query":"hook capability check","depth":5})]:
            url=(self.control_url if path=="/api/health" else f"http://127.0.0.1:{self.dashboard_port}")+path
            deadline=time.monotonic()+5
            while True:
                try:
                    req=urllib.request.Request(url,data=json.dumps(body).encode() if body else None,
                                               headers={"Content-Type":"application/json"})
                    with urllib.request.urlopen(req,timeout=2) as r:status,data=r.status,r.read().decode()
                    break
                except urllib.error.HTTPError as exc:status,data=exc.code,exc.read().decode();break
                except OSError:
                    if time.monotonic()>deadline:raise HookViolation("isolated native dashboard unavailable")
                    time.sleep(.05)
            self.native_status[path]={"status":status,"response":data}
        self.log({"kind":"native_capabilities","endpoints":self.native_status})

    def event(self,event,payload=None):
        if event not in EVENTS:raise HookViolation("unknown lifecycle event")
        if not self.engine.client or self.engine.client._proc.poll() is not None:raise HookViolation("engine died before hook event")
        value={"hook_event_name":event,"session_id":self.session,"cwd":str(self.cwd),
               "transcript_path":str(self.transcript),"source":"startup",**(payload or {})}
        context=[];blocked=False;reasons=[]
        for entry in self.events[event]:
            matcher=entry.get("matcher")
            target=value.get("source","") if event=="SessionStart" else value.get("tool_name","")
            if matcher and matcher!="*" and not re.search(matcher,target):continue
            before=len(self.errors)
            try:
                result=subprocess.run(entry["command"],shell=True,executable="/bin/bash",cwd=self.cwd,env=self.env,
                    input=json.dumps(value),text=True,capture_output=True,timeout=entry.get("timeout",30))
            except subprocess.TimeoutExpired as exc:raise HookViolation("original hook timeout: "+entry["command"]) from exc
            self.log({"kind":"hook_event","event":event,"entry":entry,"input":value,
                      "exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
            if len(self.errors)>before:raise HookViolation("hook transport/model violation: "+self.errors[-1])
            if result.returncode==2:
                blocked=True;reasons.append(result.stderr or result.stdout)
            elif result.returncode!=0:raise HookViolation(f"hook command failed {result.returncode}: {entry['command']}")
            if result.stdout.strip():
                try:
                    parsed=json.loads(result.stdout)
                    text=parsed.get("hookSpecificOutput",{}).get("additionalContext")
                    if text:context.append(text)
                    if parsed.get("decision")=="block":blocked=True;reasons.append(parsed.get("reason","hook blocked"))
                except json.JSONDecodeError:
                    # Claude command-hook contract injects successful stdout
                    # for SessionStart/UserPromptSubmit. Stop stdout is ignored.
                    if event in {"SessionStart","UserPromptSubmit"}:context.append(result.stdout.strip())
        return {"context":context,"blocked":blocked,"reasons":reasons}

    def begin_turn(self,prompt):
        with self.transcript.open("a") as f:f.write(json.dumps({"role":"user","content":prompt})+"\n")
        result=self.event("UserPromptSubmit",{"prompt":prompt})
        if result["blocked"]:raise HookViolation("UserPromptSubmit blocked: "+"\n".join(result["reasons"]))
        return self.drain_context()+result["context"]

    def drain_context(self):
        result=list(self.pending_context)
        self.pending_context.clear()
        return result

    def finish_turn(self,messages):
        # Only the visible transcript reaches product hooks. No scorer payload
        # or benchmark identifier is accepted by this method.
        with self.transcript.open("w") as f:
            for message in messages:f.write(json.dumps(message,ensure_ascii=False)+"\n")
        return self.event("Stop",{"stop_hook_active":False})

    def close(self):
        if self.server:self.server.shutdown();self.server.server_close();self.server=None
        if self.cli_server:self.cli_server.shutdown();self.cli_server.server_close();self.cli_server=None
        if self.cli_lock:self.cli_lock.close();self.cli_lock=None
        if self._cwd_scope:self._cwd_scope.cleanup();self._cwd_scope=None
