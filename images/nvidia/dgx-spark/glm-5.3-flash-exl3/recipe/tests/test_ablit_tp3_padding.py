"""Synthetic donor-sharding checks; no donor/model artifacts are needed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ablit',ROOT/'overlay/ablit_runtime.py')
ablit=importlib.util.module_from_spec(spec);spec.loader.exec_module(ablit)


class PaddingTests(unittest.TestCase):
    def apply(self,donor,local,world,rank):
        layer=SimpleNamespace(weight=torch.zeros(donor.shape[0],local))
        with patch.object(ablit,'unwrap_text_model',lambda x:x),patch.object(ablit,'walk_o_proj',lambda x:[('layers.15.self_attn.o_proj',15,layer)]),patch.object(ablit,'_tp_world',lambda:world),patch.object(ablit,'_tp_rank',lambda:rank):
            ablit.apply_transplant(object(),{15:donor},[15],False)
        return layer.weight

    def test_tp3_end_padding_and_exact_shards(self):
        for width,local in [(8,3),(9,3),(32,11),(64,22)]:
            donor=torch.arange(2*width,dtype=torch.float32).reshape(2,width)
            actual=torch.cat([self.apply(donor,local,3,r) for r in range(3)],dim=1)
            self.assertTrue(torch.equal(actual,torch.nn.functional.pad(donor,(0,3*local-width))))

    def test_tp2_unchanged(self):
        donor=torch.arange(16,dtype=torch.float32).reshape(2,8)
        self.assertTrue(torch.equal(torch.cat([self.apply(donor,4,2,r) for r in range(2)],dim=1),donor))
        with self.assertRaises(ablit.AblitError):self.apply(donor,5,2,0)

    def test_real_mismatch_rejected(self):
        for width,local in [(5,3),(10,3)]:
            with self.assertRaises(ablit.AblitError):self.apply(torch.zeros(2,width),local,3,0)


if __name__=='__main__':unittest.main()
