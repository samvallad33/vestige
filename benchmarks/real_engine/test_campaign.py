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
