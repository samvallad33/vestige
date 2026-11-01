"""Stage transitions and cleanup; transport fixtures never count as scored data."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
from .run import Campaign
from .integrity import IntegrityInvalid

class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def check_transition(self,status):
        with tempfile.TemporaryDirectory() as tmp:
            campaign=Campaign.__new__(Campaign)
            campaign.autonomous=True
            campaign.runs=Path(tmp)
            campaign.config_hash='fixture'
            campaign.names=['memconflict']
            campaign.preflight_count=0
            campaign.preflight_counts={'memconflict':10}
            campaign.solve=AsyncMock(return_value={'kill_control_passed':True})
            stages=[]
            async def benchmark(name):stages.append(campaign.preflight_for(name))
            campaign.benchmark=benchmark
            report={'benchmarks':{'memconflict':{'status':status,'ready_to_score':status!='FAIL'}}}
            with patch('benchmarks.real_engine.extra_datasets.load_memconflict',return_value=[SimpleNamespace(id='fixture')]),patch('benchmarks.real_engine.integrity.audit',return_value=report):
                if status=='FAIL':
                    with self.assertRaises(IntegrityInvalid):await campaign.pipeline('memconflict')
                else:await campaign.pipeline('memconflict')
            return stages,campaign.solve.await_count
    async def test_full_starts_immediately_after_qualified_pilot_and_real_control(self):
        stages,controls=await self.check_transition('PENDING')
        self.assertEqual(stages,[10,0]);self.assertEqual(controls,1)
    async def test_invalid_pilot_never_starts_full(self):
        stages,controls=await self.check_transition('FAIL')
        self.assertEqual(stages,[10]);self.assertEqual(controls,1)
    async def test_direct_full_path_checks_execution_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Campaign.__new__(Campaign);c.runs=Path(tmp);c.names=['locomo'];c.preflight_count=0;c.preflight_counts={}
            (c.runs/'QUALIFICATION.json').write_text('{"vestige_arm":"memory_qa_hooks"}')
            report={'benchmarks':{'locomo':{'status':'FAIL','ready_to_score':True}}}
            with patch('benchmarks.real_engine.integrity.audit',return_value=report):
                with self.assertRaises(IntegrityInvalid):await c.benchmark('locomo')

class ActualNativeProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def run_native_fixture(self,responses,kill_on_reply=False,question='What changed?'):
        import json,shutil
        from .run import REPO
        from .provider import MODEL
        class ScriptedProvider:
            # Provider-only fixture. The memory process below is the real
            # hashed child; this fixture makes no network/inference calls.
            async def chat(self,*args,**kwargs):
                reply=responses.pop(0)
                if kill_on_reply:
                    from .engine import ACTIVE_ENGINES
                    owned=[e for e in ACTIVE_ENGINES if e.run_dir.is_relative_to(root)]
                    assert len(owned)==1
                    owned[0].client._proc.kill()
                    owned[0].client._proc.wait(timeout=10)
                return {'message':reply[0],'finish_reason':reply[1],'cost':0.0,
                        'usage':{'prompt_tokens':1,'completion_tokens':1},'model':MODEL}
        with tempfile.TemporaryDirectory(dir=REPO/'runs') as tmp:
            root=Path(tmp)
            for name in ('build-manifest.json','hook-audit.json'):
                shutil.copyfile(REPO/'runs'/name,root/name)
            c=Campaign.__new__(Campaign);c.runs=root;c.config_hash='unit-actual-native';c.preflight_counts={};c.preflight_count=0
            c.manifest=json.loads((root/'build-manifest.json').read_text());c.provider=ScriptedProvider()
            item=SimpleNamespace(id='unit-native',history=['Calendar fixture: appointment moved to Monday.'],question=question,group_id='unit',category='unit')
            row=await c.solve('unit-only',item,'vestige',0,0)
            from .integrity import _native_item
            self.assertEqual(_native_item(row,root/'unit-checkpoint',root,c.manifest)['status'],'PASS')
            return row
    async def test_real_native_error_reaches_model_unchanged_then_answer(self):
        import json
        row=await self.run_native_fixture([
            ({'role':'assistant','content':None,'tool_calls':[{'id':'native-invalid','type':'function','function':{'name':'recall','arguments':json.dumps({'mode':'reason','query':'calendar fixture'})}}]},'tool_calls'),
            ({'role':'assistant','content':'No supported lookup completed.'},'stop')])
        tools=[m for m in row['transcript'] if m.get('role')=='tool']
        raw=json.loads(tools[0]['content'])
        self.assertTrue(raw['isError']);self.assertEqual(row['termination'],'answer')
    async def test_real_episode_output_limit_is_preserved_for_official_scoring(self):
        row=await self.run_native_fixture([({'role':'assistant','content':'Actual partial model output'},'length')])
        self.assertEqual(row['status'],'complete');self.assertEqual(row['termination'],'output_limit')
        self.assertEqual(row['response'],'Actual partial model output');self.assertIsNone(row['score'])

    async def test_engine_death_during_final_limited_reply_still_aborts(self):
        from .engine import VestigeMCPError
        with self.assertRaisesRegex(VestigeMCPError,'not running'):
            await self.run_native_fixture([({'role':'assistant','content':'Partial'},'length')],kill_on_reply=True)

    async def test_initial_context_limit_never_invents_model_answer_or_call(self):
        from .provider import MODEL
        row=await self.run_native_fixture([],question='long '*40000)
        self.assertEqual(row['model'],MODEL);self.assertEqual(row['usage'],[])
        self.assertEqual(row['response'],'');self.assertEqual(row['termination'],'context_limit')
