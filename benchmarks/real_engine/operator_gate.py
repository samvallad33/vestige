"""The owner's exact full Operator gate, isolated and executed before effects.

This module contains no safety-rule implementation. It copies verified source
and companions, uses the gate's supported disabled_rules configuration only for
independently audited benchmark-specific rules, executes its actual hook CLI,
and records original receipts. A stop is an observed stop, never a safety score.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid

FULL_GATE = Path.home()/"Developer/vestige-v6/operator-lite/operator-gate.py"
FULL_GATE_SHA256 = "8aac211ee017c070a3e4f6d29234728de059dd825ca38c72000231338089a64a"
FULL_GATE_BYTES = 645044
INSTALLED_COMPANIONS = Path.home()/".operator/gate"
COMPANIONS = ("lite_core.py","sealed.py","briefing.py","board.py","letter.py","almost_forgot.py","onboard.py","replay.py")
LITE_SHA256 = "e06bc5f0022f148f53c2dfc30fecb09bd8e6357b4c1b53c058ef88cc81c03f97"


class OperatorGateError(RuntimeError):pass


class OperatorBlocked(OperatorGateError):
    def __init__(self,verdict):
        self.verdict=verdict
        super().__init__(verdict.get("stderr") or "native full Operator gate stopped the tool call")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"))


def inspect_full_source(path=FULL_GATE):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise OperatorGateError("full gate source must be a regular file")
    body=path.read_bytes()
    if len(body)!=FULL_GATE_BYTES or hashlib.sha256(body).hexdigest()!=FULL_GATE_SHA256:
        raise OperatorGateError("full Operator source differs from the owner-selected pin")
    module=ast.parse(body.decode())
    assignment=next(n for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="RULES" for t in n.targets))
    rules=ast.literal_eval(assignment.value)
    return {"bytes":len(body),"sha256":FULL_GATE_SHA256,"registered_rule_ids":sorted(rules),
            "total_registered_rule_count":len(rules),"numeric_rule_count":sum(k.removeprefix("OP-").isdigit() for k in rules),
            "default_modes":{k:value[1].lower() for k,value in rules.items()}}


def canonical_benchmark(name):
    name=str(name or "unknown").lower().replace("-","_")
    if name.startswith("memoryagentbench_"):return "memoryagentbench"
    return {"terminalbench":"terminal_bench","agent_dojo":"agentdojo","injec_agent":"injecagent"}.get(name,name)


class OperatorGate:
    def __init__(self,repo,operator_home,home,benchmark,*,audit_path=None,source_path=FULL_GATE,companion_dir=INSTALLED_COMPANIONS):
        self.repo=Path(repo).resolve();self.operator_home=Path(operator_home).resolve();self.home=Path(home).resolve()
        self.benchmark=canonical_benchmark(benchmark)
        self.audit_path=Path(audit_path or self.repo/"runs/operator-rule-audit.json")
        self.source=Path(source_path);self.companion_dir=Path(companion_dir)
        self.original=inspect_full_source(self.source)
        self.offsets={};self.lock=threading.RLock();self.identity={};self.prepared=False
        self.log_path=self.operator_home/"operator-calls.jsonl"

    def log(self,value):
        with self.log_path.open("a") as f:f.write(canonical(value)+"\n");f.flush();os.fsync(f.fileno())

    def prepare(self):
        if not self.audit_path.is_file():raise OperatorGateError("full Operator rule audit is missing; no tool effect permitted")
        audit=json.loads(self.audit_path.read_text())
        if audit.get("complete") is not True:raise OperatorGateError("full Operator rule audit is incomplete")
        pin=audit.get("source_pins",{}).get("operator_gate",{})
        if pin.get("sha256")!=FULL_GATE_SHA256:raise OperatorGateError("audited full gate pin mismatch")
        if sorted(audit.get("registered_rule_ids",[]))!=self.original["registered_rule_ids"]:
            raise OperatorGateError("Operator rule audit does not cover the complete registry")
        disabled=sorted(set(audit.get("disabled_by_benchmark",{}).get(self.benchmark,[])))
        decisions={row["id"]:row for row in audit.get("rules",[])}
        if sorted(decisions)!=self.original["registered_rule_ids"]:
            raise OperatorGateError("Operator audit does not classify every registered rule")
        for rule in disabled:
            decision=decisions.get(rule,{})
            if decision.get("classification") not in {"CASE_TUNED","case_tuned","benchmark_tuned"} or self.benchmark not in decision.get("disable_for_benchmarks",[]):
                raise OperatorGateError("unsupported/unreviewed rule disable: "+rule)
        gate_dir=self.operator_home/"gate"
        if gate_dir.exists():raise OperatorGateError("worker gate source directory must be fresh")
        for historical in ("receipts","permits","approvals"):
            p=self.operator_home/historical
            if p.exists() and any(p.iterdir()):raise OperatorGateError("worker Operator home contains historical "+historical)
        gate_dir.mkdir(mode=0o700)
        copied={}
        for name,src in [("operator-gate.py",self.source)]+[(name,self.companion_dir/name) for name in COMPANIONS]:
            if src.is_symlink() or not src.is_file():raise OperatorGateError("required native gate companion missing/nonregular: "+name)
            audit_key="operator_gate" if name=="operator-gate.py" else Path(name).stem
            audited=audit.get("source_pins",{}).get(audit_key,{})
            if audited.get("sha256")!=sha(src):raise OperatorGateError("native gate companion changed after audit: "+name)
            dest=gate_dir/name;shutil.copyfile(src,dest);os.chmod(dest,0o700 if name=="operator-gate.py" else 0o600)
            if sha(src)!=sha(dest):raise OperatorGateError("native gate copy differs: "+name)
            copied[name]={"original_sha256":sha(src),"effective_sha256":sha(dest),"bytes":dest.stat().st_size}
        if copied["lite_core.py"]["effective_sha256"]!=LITE_SHA256:
            raise OperatorGateError("full gate's pinned Lite core is missing or mismatched")
        source_policy=self.operator_home/"commitments.json"
        before=source_policy.read_bytes();config=json.loads(before)
        inherited=set(config.get("disabled_rules",[]))
        if inherited-set(disabled):raise OperatorGateError("worker inherited unaudited disabled rules")
        if disabled:
            config["disabled_rules"]=disabled
            source_policy.write_text(canonical(config)+"\n");os.chmod(source_policy,0o600)
        self.gate_path=gate_dir/"operator-gate.py"
        self.identity={"version":"6.2.0","benchmark":self.benchmark,"source":self.original,"copied_sources":copied,
            "original_gate_sha256":FULL_GATE_SHA256,"effective_gate_sha256":sha(self.gate_path),
            "original_commitments_sha256":hashlib.sha256(before).hexdigest(),"effective_commitments_sha256":sha(source_policy),
            "disabled_rule_ids":disabled,"config_transform":"supported commitments.json.disabled_rules only",
            "audit_sha256":sha(self.audit_path),"registered_only_rule_ids":audit.get("registered_only_rule_ids",[]),
            "emitter_present_rule_ids":audit.get("emitter_present_rule_ids",[]),"integrity":"reference_digest_not_signature"}
        # Verify the exact gate can actually import its pinned Lite core under
        # this isolated HOME, without executing any proposed tool action.
        probe_script="import importlib.util,json,sys; s=importlib.util.spec_from_file_location('native_operator_probe',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); c=m.lite_core(); print(json.dumps({'loaded':c is not None,'status':m._LITE_CORE.get('status'),'version':getattr(c,'VERSION',None)}))"
        probe_env={"HOME":str(self.home),"OPERATOR_HOME":str(self.operator_home),"PATH":"/usr/bin:/bin"}
        probe=subprocess.run([sys.executable,"-c",probe_script,str(self.gate_path)],env=probe_env,cwd=self.home,text=True,capture_output=True,timeout=15,check=True)
        core_status=json.loads(probe.stdout)
        if core_status.get("loaded") is not True:raise OperatorGateError("pinned Lite component failed its native import")
        self.identity["lite_core_runtime"]=core_status
        self.log({"kind":"operator_prepared",**self.identity})
        self.prepared=True
        return self.identity

    def _new_receipts(self):
        result=[];directory=self.operator_home/"receipts"
        if not directory.exists():return result
        for path in sorted(directory.glob("????-??-??.jsonl")):
            start=self.offsets.get(str(path),0)
            with path.open("rb") as f:
                f.seek(start)
                for line in f:
                    if not line.endswith(b"\n"):raise OperatorGateError("native receipt append is incomplete")
                    if line.strip():result.append(json.loads(line))
                self.offsets[str(path)]=f.tell()
        return result

    def check(self,tool_name,tool_input,*,cwd,session_id,call_id=None,env=None):
        if not self.prepared:raise OperatorGateError("native full gate is not prepared")
        with self.lock:
            if sha(self.gate_path)!=self.identity["effective_gate_sha256"]:raise OperatorGateError("worker full gate changed after pin")
            if sha(self.operator_home/"commitments.json")!=self.identity["effective_commitments_sha256"]:
                raise OperatorGateError("worker rule policy changed after pin")
            cid=call_id or "native-gate-"+uuid.uuid4().hex
            payload={"hook_event_name":"PreToolUse","tool_name":str(tool_name),"tool_input":tool_input,
                "cwd":str(cwd),"session_id":str(session_id),"tool_use_id":cid}
            environment={"HOME":str(self.home),"PATH":"/usr/bin:/bin:/usr/sbin:/sbin","OPERATOR_HOME":str(self.operator_home),
                "OPERATOR_GATE_MODE":"enforce","XDG_CONFIG_HOME":str(self.home/".config"),
                "XDG_DATA_HOME":str(self.home/".local/share"),"XDG_CACHE_HOME":str(self.home/".cache")}
            if env:
                for key in ("TMPDIR","GIT_CEILING_DIRECTORIES","GIT_CONFIG_NOSYSTEM","GIT_CONFIG_GLOBAL"):
                    if key in env:environment[key]=env[key]
            # Read only receipts created by this isolated native invocation.
            old=self._new_receipts()
            if old:self.log({"kind":"operator_out_of_band_receipts","receipts":old})
            process=subprocess.run([sys.executable,str(self.gate_path),"hook","--source","codex"],
                input=canonical(payload),text=True,capture_output=True,cwd=cwd,env=environment,timeout=30)
            receipts=self._new_receipts()
            if not receipts:raise OperatorGateError("full native gate returned without a receipt")
            errors=[row for row in receipts if row.get("decision")=="GATE_ERROR"]
            if process.returncode not in {0,2} or errors:
                self.log({"kind":"operator_error","payload":payload,"exit_code":process.returncode,"stdout":process.stdout,"stderr":process.stderr,"receipts":receipts})
                raise OperatorGateError("full native Operator gate failed; no fallback effect permitted")
            fired=set()
            for row in receipts:
                fired.update(row.get("commitments") or [])
                if row.get("commitment"):fired.add(row["commitment"])
            verdict={"allowed":process.returncode==0,"exit_code":process.returncode,"tool":tool_name,"call_id":cid,
                "fired_rule_ids":sorted(fired),"receipts":receipts,"stdout":process.stdout,"stderr":process.stderr,
                "original_gate_sha256":FULL_GATE_SHA256,"effective_gate_sha256":self.identity["effective_gate_sha256"],
                "disabled_rule_ids":self.identity["disabled_rule_ids"],"interpretation":"actual native gate outcome; legitimacy/safety not inferred"}
            self.log({"kind":"operator_verdict","payload":payload,**verdict})
            return verdict

    def require(self,tool_name,tool_input,**kwargs):
        verdict=self.check(tool_name,tool_input,**kwargs)
        if not verdict["allowed"]:raise OperatorBlocked(verdict)
        return verdict
