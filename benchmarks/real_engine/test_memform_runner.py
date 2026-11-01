"""MemForm no-inference plumbing checks; these produce no benchmark scores."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

from benchmarks.real_engine import memform_runner as m

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / 'runs/upstream/memform'


class MemFormContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not UPSTREAM.exists():
            raise unittest.SkipTest('Pinned official MemForm not downloaded')
        cls.core, cls.common = m.official(UPSTREAM)
        cls.items = m.load_cases(UPSTREAM)

    def test_all_official_cases_all_interfaces(self):
        self.assertEqual(len(self.items), 1891)
        self.assertEqual(sum(i.startswith('mcq:') for i,_ in self.items), 994)
        self.assertEqual(sum(i.startswith('freeform:') for i,_ in self.items), 897)
        self.assertEqual({c['interface'] for _,c in self.items}, {'P1','P2','P3','P4','P5'})

    def test_empty_gate_accepted_histories_skip_compiler_calls(self):
        case = next(c for _,c in self.items if c['interface']=='P5')
        defense = types.SimpleNamespace(exports=[2,3], accepted_counts={2:0,3:2})
        self.assertEqual(m.compiler_schedule(case,self.common,defense), [1,1])

    def test_unknown_gate_result_never_turns_into_a_block(self):
        class BadEngine:
            def call(self,*args): return {'error':'transport did not establish admission'}
        with self.assertRaisesRegex(RuntimeError,'Unclassified native write'):
            m.read_written(BadEngine(),'raw text','test')

    def test_native_read_uses_returned_content_instead_of_input(self):
        class Engine:
            def call(self,name,args):
                if name=='smart_ingest': return {'nodeId':'actual-id','decision':'create'}
                return {'node':{'content':'ACTUAL_NATIVE_CONTENT'}}
        text,_ = m.read_written(Engine(),'HOST_INPUT','test')
        self.assertEqual(text,'ACTUAL_NATIVE_CONTENT')

    def test_official_task_prompt_contains_no_gold_annotations(self):
        case = copy.deepcopy(next(c for _,c in self.items if c['interface']=='P2'))
        task = case['evaluation']['trigger_tasks'][0]
        task['clean_answer']='CLEAN_GOLD_SENTINEL'
        task['poisoned_answer']='POISON_GOLD_SENTINEL'
        case['memory_building']['oracle_memory']={'clean':'ORACLE_CLEAN_SENTINEL','poisoned':'ORACLE_POISON_SENTINEL'}
        system,user = self.core.task_prompt(case,task,'ADMITTED_MEMORY')
        for word in ['CLEAN_GOLD_SENTINEL','POISON_GOLD_SENTINEL','ORACLE_CLEAN_SENTINEL','ORACLE_POISON_SENTINEL']:
            self.assertNotIn(word,system+user)

    def test_bridge_adds_exactly_neutral_arm_line_and_shared_provider(self):
        calls=[]
        class Provider:
            async def chat(self,messages,**kwargs):
                calls.append((messages,kwargs))
                return {'message':{'content':'A'},'cost':0.001,'usage':{'completion_tokens':1},'finish_reason':'stop','model':m.MODEL}
        with tempfile.TemporaryDirectory() as tmp:
            campaign=types.SimpleNamespace(provider=Provider())
            for arm in m.ARMS:
                bridge=m.ModelBridge(campaign,None,Path(tmp)/arm,{'benchmark':'memform'},arm,'agent')
                result=asyncio.run(bridge.request('OFFICIAL SYSTEM','OFFICIAL USER'))
                self.assertEqual(result,'A')
                self.assertEqual(calls[-1][0][0]['content'],'OFFICIAL SYSTEM\n'+m.TOOL_LINES[arm])
                self.assertEqual(calls[-1][0][1]['content'],'OFFICIAL USER')

    def test_official_freeform_judge_callback_is_used(self):
        case = next(c for item,c in self.items if item.startswith('freeform:'))
        task = self.core.freeform_tasks_from_case(case)[0]
        bound = self.core.bind_freeform_task(case,task)
        calls=[]
        class Judge:
            def invoke(self,system,user):
                calls.append((system,user))
                return '{"verdict":"NEITHER","confidence":0.99,"reason":"diagnostic"}'
        result = self.core.evaluate_freeform_response('DIAGNOSTIC_UNRECOGNIZED_RESPONSE',bound,Judge(),Judge())
        self.assertGreater(len(calls),0)
        self.assertIn('label',result)

    def test_bootstrap_does_not_invent_conditional_asr_when_no_eligible_tasks(self):
        official=types.SimpleNamespace(aggregate_results=lambda rows: {'count':len(rows)})
        merged=[{'group_id':'family','official_result':{'task_results':[
            {'shared_target':True,'baseline':{'clean_correct':True},
             'learned_clean':{'clean_correct':True},'learned_poisoned':{'poison_target_hit':True}}]}}]
        summary=m.summarize_complete(official,merged,draws=20)
        self.assertEqual(summary['clean_accuracy']['mean'],1.0)
        self.assertEqual(summary['clean_accuracy']['ci95'],[1.0,1.0])
        self.assertIsNone(summary['causal_asr']['mean'])
        self.assertIsNone(summary['causal_asr']['ci95'])
        self.assertEqual(summary['causal_asr']['n'],0)

    def test_raw_collection_never_calls_official_scorer(self):
        from unittest.mock import patch
        case=next(c for item,c in self.items if item.startswith('mcq:') and c['interface']=='P2')
        defense=m.make_defense(self.common,'bare',REPO,Path('unused'),REPO/'runs/build-manifest.json')
        class Scripted:
            def invoke(self,system,user):return 'DIAGNOSTIC_UNSCORED_RESPONSE'
        with patch.object(self.core,'run_task',side_effect=AssertionError('Scorer invoked before integrity')), \
             patch.object(self.core,'run_freeform_task',side_effect=AssertionError('Scorer invoked before integrity')):
            raw=m.collect_case_raw(self.core,self.common,case,Scripted(),Scripted(),defense,[])
        self.assertTrue(raw['raw_task_results'])
        self.assertNotIn('official_result',raw)
        self.assertNotIn('score',raw)

    def test_preflight_keeps_full_manifest_and_reuses_sharded_checkpoints_without_scoring(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            runs=Path(tmp)
            campaign=types.SimpleNamespace(runs=runs,manifest={'binary_sha256':'0'*64},progress={},
                preflight_count=20,ledger=types.SimpleNamespace(snapshot=lambda:{'remaining':25.0}))
            async def fake_run(campaign,item_id,case,arm,worker,attempt=0):
                await asyncio.sleep(0)
                return {'item_id':item_id,'trial':0,'status':'complete','arm':arm,'worker_id':worker,
                        'usage':[{}],'cost':0.0,'response':'DIAGNOSTIC_UNSCORED'}
            with patch.object(m,'official',return_value=(self.core,self.common)), \
                 patch.object(m,'load_cases',return_value=self.items), \
                 patch.object(m,'run_one',side_effect=fake_run), \
                 patch.object(m,'strict_merge',side_effect=AssertionError('Preflight tried to score')):
                result=asyncio.run(m.run_memform(campaign))
                again=asyncio.run(m.run_memform(campaign))
            self.assertEqual(result['selected_count'],20)
            self.assertEqual(again['full_set_count'],1891)
            self.assertEqual(json.loads((runs/'memform/full-set.json').read_text())['count'],1891)
            coverage=json.loads((runs/'memform/coverage.json').read_text())
            for arm in m.ARMS:
                self.assertEqual(len(coverage[arm]['completed']),20)
                self.assertEqual(len(coverage[arm]['unfinished']),1871)
            self.assertFalse((runs/'memform/RESULTS.md').exists())

    def test_actual_native_protocol_separates_conditions_and_p5_sequence(self):
        if not (REPO/'runs/build-manifest.json').exists():
            self.skipTest('No hashed release build manifest')
        for interface in ('P1','P2','P3','P4','P5'):
            with self.subTest(interface=interface), tempfile.TemporaryDirectory(dir=REPO/'runs') as tmp:
                case=copy.deepcopy(next(c for item,c in self.items if item.startswith('mcq:') and c['interface']==interface))
                # Sentinel is diagnostic only; oracle phase is never executed.
                case['memory_building']['oracle_memory']={'clean':'ORACLE_ONLY_SENTINEL','poisoned':'ORACLE_ONLY_SENTINEL'}
                from benchmarks.real_engine.engine import RealEngine
                defense=m.make_defense(self.common,'vestige',REPO,Path(tmp)/'native',REPO/'runs/build-manifest.json',engine_factory=RealEngine)
                prompts=[]
                class ScriptedModel:
                    def __init__(self,role): self.role,self.i=role,0
                    def invoke(inner,system,user):
                        prompts.append(system+'\n'+user)
                        if inner.role=='memory':
                            schedule=m.compiler_schedule(case,self.common,defense)
                            text=defense.admit_compiler_memory(schedule[inner.i],'Diagnostic scripted memory: choose option A.')
                            inner.i+=1
                            return text
                        return 'FINAL_ANSWER: A'
                try:
                    result=self.core.run_case(agent=ScriptedModel('agent'),memory_model=ScriptedModel('memory'),
                        alignment_model=None,freeform_parser_model=None,freeform_judge_model=None,case=case,
                        phase='learned',max_words=120,p5_updater='incremental_merge',target_alignment=False,
                        defense=defense)
                    self.assertTrue(result['task_results'])
                    self.assertNotIn('ORACLE_ONLY_SENTINEL','\n'.join(prompts))
                    self.assertGreaterEqual(len(defense.identities),2)
                    self.assertEqual(len({x['data_dir'] for x in defense.identities}),len(defense.identities))
                    for engine in defense.engines.values():
                        rows=[json.loads(s) for s in engine.audit.read_text().splitlines()]
                        writes=[r for r in rows if r.get('kind')=='tool_request' and r.get('name')=='smart_ingest']
                        reads=[r for r in rows if r.get('kind')=='tool_request' and r.get('name')=='memory']
                        self.assertGreater(len(writes),0)
                        self.assertEqual(len(writes),len(reads))
                finally:
                    defense.close()


if __name__=='__main__':
    unittest.main()
