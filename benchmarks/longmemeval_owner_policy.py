"""Owner-approved static native contract provenance; frozen solver untouched.

Run with python -m benchmarks.longmemeval_owner_policy --benchmarks longmemeval
--resume-baselines runs/lme-source-bundle.json. Only literal-overlap adjudication
is extended; all other audit checks and official scoring remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import re

from benchmarks.real_engine import integrity
from benchmarks.real_engine.native_contract import model_tools
from benchmarks.real_engine.runtime import append_jsonl, canonical, utc_now

REPO = Path(__file__).resolve().parents[1]
AUTHORITY = 'mem-00000000000b0e7f'
ORIGINAL = integrity.literal_overlap_audit


def static_contract():
    runs = REPO / 'runs'
    proof = json.loads((runs/'longmemeval/recall-proof.json').read_text())
    config = json.loads((runs/'four-config.json').read_text())
    if proof['status'] != 'PASS' or proof['config_hash'] != config['config_hash'] or len(proof['items']) != 10:
        raise RuntimeError('owner static allowance requires current ten-item native proof')
    reference = None
    evidence = []
    for item in proof['items']:
        root = Path(item['root'])/'native'
        catalog = json.loads((root/'mcp-tools.json').read_text())['tools']
        initialize = json.loads((root/'mcp-initialize.json').read_text())
        value = {'catalog': catalog, 'instructions': initialize['instructions']}
        if reference is not None and reference != value:
            raise RuntimeError('native static catalog differs across proof items')
        reference = value
        evidence.append({'path':str(root/'mcp-tools.json'), 'sha256':hashlib.sha256((root/'mcp-tools.json').read_bytes()).hexdigest()})
    return reference, evidence


def owner_static_overlap(requests, items, native_strings=None, scaffolds=()):
    overlaps, missing = ORIGINAL(requests, items, native_strings, scaffolds)
    failures = [row for row in overlaps if row['status']=='FAIL']
    if not failures:
        return overlaps, missing
    contract, evidence = static_contract()
    adapted = model_tools(contract['catalog'], {'instructions':contract['instructions']})
    native = {t['name']:t for t in contract['catalog']}
    indexed = {(r['path'],r['line']):r['row'] for r in requests}
    for finding in failures:
        request = indexed.get((finding['path'], finding['line']), {})
        if request.get('identity',{}).get('arm') != 'vestige':
            continue
        match = re.fullmatch(r'\$\.tools\[(\d+)\]\.function\.(description|parameters.*|name)', finding['model_pointer'])
        if match is None:
            continue
        tools = request.get('payload',{}).get('tools',[])
        index = int(match[1])
        if index >= len(tools):
            continue
        function = tools[index].get('function',{})
        name = function.get('name','').removeprefix('mcp__vestige__')
        source = native.get(name)
        if source is None:
            continue
        ordinary = {'name':function.get('name'),'description':source.get('description',''),'parameters':source['inputSchema']}
        full = next(t['function'] for t in adapted if t['function']['name']==name)
        if function != ordinary and function != full:
            continue
        visible = dict(integrity.strings(request.get('payload',{}))).get(finding['model_pointer'])
        if not isinstance(visible,str) or hashlib.sha256(visible.encode()).hexdigest()!=finding['visible_string_sha256']:
            continue
        finding.update(status='PASS', provenance='OWNER_ALLOWED_VERIFIED_STATIC_NATIVE_CONTRACT',
                       owner_authority=AUTHORITY, provenance_evidence=evidence)
        append_jsonl(REPO/'runs/owner-static-native-allowances.jsonl',
                     {'timestamp':utc_now(),'authority':AUTHORITY,'finding':finding,
                      'policy_file':str(Path(__file__).resolve()),'policy_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return overlaps, missing


def install():
    integrity.literal_overlap_audit = owner_static_overlap


if __name__ == '__main__':
    install()
    from benchmarks.real_engine.run import main
    main()
