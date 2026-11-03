"""Check gold isolation, official split coverage and unchanged scorer bridging."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from . import datasets as d
except ImportError:
    import datasets as d

ROOT = Path(__file__).resolve().parents[2]


class DatasetContracts(unittest.TestCase):
    def test_solver_record_has_no_scorer_annotations(self):
        item = d.BenchmarkItem('q', 'c', ['user: ordinary history'], 'A question', '1',
                               {'answer': 'GOLD_SENTINEL', 'evidence': ['GOLD_EVIDENCE']})
        text = json.dumps(item.solver_record())
        self.assertNotIn('GOLD_SENTINEL', text)
        self.assertNotIn('GOLD_EVIDENCE', text)
        self.assertEqual(set(item.solver_record()), {'id','group_id','history','question','category'})

    def test_longmemeval_role_whitelist_drops_answer_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data' / 'longmemeval_s_cleaned.json'
            path.parent.mkdir()
            path.write_text(json.dumps([{
                'question_id':'q_abs','question_type':'multi-session','question':'Q?',
                'question_date':'2023-01-03','answer':'GOLD_SENTINEL',
                'answer_session_ids':['GOLD_SESSION'],
                'haystack_dates':['2023-01-02'], 'haystack_sessions':[[
                    {'role':'user','content':'USER_CONTENT','has_answer':True},
                    {'role':'assistant','content':'ASSISTANT_CONTENT','evidence':'GOLD_EVIDENCE'}]]}]))
            with patch.dict(d.SOURCES['longmemeval'], {'data_sha256':d.sha256(path)}), \
                 patch.object(d, '_unique', side_effect=lambda items, expected: items):
                item = d.load_longmemeval(tmp)[0]
            payload = json.dumps(item.solver_record())
            self.assertIn('USER_CONTENT', payload)
            self.assertIn('ASSISTANT_CONTENT', payload)
            self.assertIn('2023-01-02', payload)
            self.assertIn('2023-01-03', item.question)
            for forbidden in ['GOLD_SENTINEL','GOLD_SESSION','GOLD_EVIDENCE','has_answer']:
                self.assertNotIn(forbidden, payload)

    def test_hash_mismatch_fails_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data' / 'locomo10.json'
            path.parent.mkdir()
            path.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                d.load_locomo(tmp)

    def test_merged_prediction_coverage_rejects_missing_extra_duplicate(self):
        item = d.BenchmarkItem('q','c',[],'Q?','1',{})
        for items, predictions in [([item],{}), ([item],{'q':'A','extra':'A'}),
                                   ([item,item],{'q':'A'})]:
            with self.assertRaises(ValueError):
                d._check_predictions(items, predictions)

    def test_longmemeval_bridge_uses_official_prompt_and_shared_callback(self):
        root = ROOT / 'runs' / 'upstream' / 'longmemeval'
        if not root.exists():
            self.skipTest('Official pinned checkout not downloaded')
        item = d.BenchmarkItem('q_abs','q_abs',[],'Question date: D\nQ?', 'multi-session',
                              {'question':'Q?', 'question_type':'multi-session','answer':'gold'})
        calls = []
        async def callback(payload):
            calls.append(payload)
            return {'response':' Yes ', 'usage':{'input_tokens':10}, 'cost':0.001}
        scores, logs = asyncio.run(d.judge_longmemeval(root,[item],{'q_abs':'unsure'},callback))
        self.assertEqual(scores,[1.0])
        self.assertIn('unanswerable question', calls[0]['user'])
        self.assertEqual(calls[0]['user'], d.official_judge_prompt(root,item,'unsure'))
        self.assertEqual(logs[0]['cost'],0.001)

    def test_official_locomo_all_questions_categories_and_scorer(self):
        root = ROOT / 'runs' / 'upstream' / 'locomo'
        if not root.exists():
            self.skipTest('Official pinned checkout not downloaded')
        items = d.load_locomo(root)
        self.assertEqual(len(items),1986)
        self.assertEqual(len({i.group_id for i in items}),10)
        self.assertEqual({i.category for i in items},{'1','2','3','4','5'})
        five = next(i for i in items if i.category=='5')
        self.assertNotIn('adversarial_answer',json.dumps(five.solver_record()))
        self.assertEqual(d.score_locomo(root,[five],{five.id:'No information available'}),[1.0])
        self.assertEqual(d.score_locomo(root,[five],{five.id:'made up answer'}),[0.0])

    def test_official_longmemeval_all500(self):
        root = ROOT / 'runs' / 'upstream' / 'longmemeval'
        if not (root / 'data' / 'longmemeval_s_cleaned.json').exists():
            self.skipTest('Official pinned dataset not downloaded')
        items = d.load_longmemeval(root)
        self.assertEqual(len(items),500)
        self.assertEqual(len({i.id for i in items}),500)
        self.assertEqual(len({i.category for i in items}),6)
        self.assertEqual(sum('_abs' in i.id for i in items),30)


if __name__ == '__main__':
    unittest.main()
