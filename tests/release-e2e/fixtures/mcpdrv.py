import json, os, subprocess, time, socket, uuid, sys
from urllib.request import urlopen, Request
def port():
    s=socket.socket(); s.bind(("127.0.0.1",0)); p=s.getsockname()[1]; s.close(); return p
class M:
    def __init__(s,B,d,mode="risk_gated",dash=True):
        s.dash=port(); s.tok=uuid.uuid4().hex*2
        env={k:v for k,v in os.environ.items() if not k.startswith("VESTIGE_")}
        env.update(VESTIGE_DASHBOARD_ENABLED="true" if dash else "false",VESTIGE_DASHBOARD_PORT=str(s.dash),VESTIGE_AUTH_TOKEN=s.tok,VESTIGE_HTTP_ENABLED="0",VESTIGE_AUTOPILOT_ENABLED="0",RUST_LOG="warn",HOME=d)
        if mode: open(f"{d}/review_mode.json","w").write(json.dumps({"mode":mode}))
        s.p=subprocess.Popen([B,"--data-dir",d],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(f"{d}.err","w"),bufsize=0)
        s.API=f"http://127.0.0.1:{s.dash}"; s.seq=0
        if dash:
            for _ in range(300):
                try: urlopen(s.API+"/api/stats",timeout=1); break
                except OSError: time.sleep(0.1)
        s.rpc("initialize",{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"e2e","version":"1"}})
    def rpc(s,m,pa):
        s.seq+=1
        s.p.stdin.write((json.dumps({"jsonrpc":"2.0","id":s.seq,"method":m,"params":pa})+"\n").encode()); s.p.stdin.flush()
        while True:
            l=s.p.stdout.readline()
            if not l: raise RuntimeError("server exited")
            r=json.loads(l)
            if r.get("id")==s.seq: return r
    def tool(s,n,a):
        r=s.rpc("tools/call",{"name":n,"arguments":a}); res=r.get("result",r)
        return res.get("structuredContent", res)
    def http(s,method,path,body=None):
        req=Request(s.API+path,method=method,data=json.dumps(body).encode() if body is not None else None,headers={"Authorization":f"Bearer {s.tok}","Content-Type":"application/json"})
        try:
            with urlopen(req,timeout=20) as r: return r.status, json.load(r)
        except Exception as e: return getattr(e,"code",None), (e.read().decode() if hasattr(e,"read") else str(e))
    def close(s):
        s.p.stdin.close(); s.p.wait(timeout=20)
