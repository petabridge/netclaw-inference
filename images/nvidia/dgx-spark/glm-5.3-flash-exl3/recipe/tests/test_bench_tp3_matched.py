"""Verify decode timing/export semantics with synthetic SSE only."""
import io
import json
import unittest
from unittest.mock import patch
import bench_tp3_matched as bench


class StreamTests(unittest.TestCase):
    def stream(self,done=True,cap=512,reasoning=False):
        events=[{'choices':[{'delta':{'content':'one'}}]}, {'choices':[{'delta':{'content':'two'}}]},
                {'choices':[{'delta':{},'finish_reason':'length'}],'usage':{'prompt_tokens':10,'completion_tokens':cap,'prompt_tokens_details':{'cached_tokens':0}}}]
        if reasoning:events[0]['choices'][0]['delta']={'reasoning_content':'unexpected'}
        return io.BytesIO((''.join('data: '+json.dumps(x)+'\n\n' for x in events)+('data: [DONE]\n\n' if done else '')).encode())

    def test_metric_excludes_ttft_and_stream_tail(self):
        with patch.object(bench.urllib.request,'urlopen',return_value=self.stream()),patch.object(bench.time,'monotonic',side_effect=[0,1,3,4]):
            out=bench.request('http://localhost/v1','fixture','synthetic',512,'')
        self.assertEqual(out['decode_tok_s'],511/2)
        self.assertEqual(out['ttft_s'],1)
        self.assertEqual(out['e2e_s'],4)
        self.assertEqual(out['cached_tokens'],0)

    def test_incomplete_or_wrong_length_refused(self):
        for done,cap in [(False,512),(True,511)]:
            with self.subTest(done=done,cap=cap),patch.object(bench.urllib.request,'urlopen',return_value=self.stream(done,cap)),self.assertRaises(RuntimeError):
                bench.request('http://localhost/v1','fixture','synthetic',512,'')

    def test_reasoning_refused(self):
        with patch.object(bench.urllib.request,'urlopen',return_value=self.stream(reasoning=True)),self.assertRaises(RuntimeError):
            bench.request('http://localhost/v1','fixture','synthetic',512,'')


if __name__=='__main__':unittest.main()
