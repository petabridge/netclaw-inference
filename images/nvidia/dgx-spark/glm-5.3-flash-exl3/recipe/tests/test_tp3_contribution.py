"""Offline tests for artifact preparation and rank integration; no engine calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'extensions/cooperative_moe/tp3'))
import manifest
import prepare_profile
import stage_bundle
import select_policy


class BundleTests(unittest.TestCase):
    def bundle(self, root):
        source = ROOT / 'extensions/cooperative_moe/tp3'
        for name in ('runtime.py', 'PROVENANCE.json'):
            (root / name).write_bytes((source / name).read_bytes())
        (root / 'cooperative_moe.so').write_bytes(b'CPU fixture, not a library')
        (root / 'toolchain.txt').write_text('CPU fixture')
        policy = json.loads((source / 'dispatch_policy.json').read_text())
        policy['native_sha256'] = manifest.digest(root / 'cooperative_moe.so')
        (root / 'dispatch_policy.json').write_text(json.dumps(policy))
        manifest.create(root)

    def test_stage_manifest_only_and_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'source', Path(tmp)/'dest'
            source.mkdir(); self.bundle(source)
            (source / 'unrelated-private-file').write_text('must not copy')
            stage_bundle.stage(source,dest)
            self.assertFalse((dest/'unrelated-private-file').exists())
            manifest.verify_artifacts(dest)
            overlay = prepare_profile.prepare(ROOT/'overlay/exl3.py', dest)
            compile(overlay.read_text(), str(overlay), 'exec')
            self.assertTrue(overlay.read_text().splitlines()[-1].startswith('_coop_setup["install"]'))
            with self.assertRaises(FileExistsError):
                prepare_profile.prepare(ROOT/'overlay/exl3.py', dest)

    def test_corruption_and_unmeasured_policy_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.bundle(root)
            (root/'cooperative_moe.so').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):manifest.verify_artifacts(root)
            manifest.create(root)
            with self.assertRaisesRegex(RuntimeError,'reprofile'):manifest.verify_artifacts(root)

    def test_manifest_escape_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bundle';root.mkdir();self.bundle(root)
            outside=Path(tmp)/'outside';outside.write_text('fixture')
            data=json.loads((root/'manifest.json').read_text())
            data['files']['../outside']=manifest.digest(outside)
            (root/'manifest.json').write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):manifest.verify_artifacts(root)

    def test_unknown_exl3_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            stock=Path(tmp)/'stock.py';stock.write_text('pass')
            with self.assertRaisesRegex(ValueError,'source changed'):prepare_profile.prepare(stock,Path(tmp))

    def test_abi2_missing_manifest_refused_before_cache_fallback(self):
        source = (ROOT / 'start-tp3.sh').read_text()
        helpers = source.split('_glm53_coop_src_dir() {', 1)[1].split('_tp3_stage_coop_runtime() {', 1)[0]
        script = '_glm53_coop_src_dir() {' + helpers + '\n_glm53_coop_src_dir\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, cache = root/'bundle', root/'cache'
            bundle.mkdir(); (cache/'cooperative_moe').mkdir(parents=True)
            self.bundle(bundle)
            overlay = prepare_profile.prepare(ROOT/'overlay/exl3.py', bundle)
            self.bundle(cache/'cooperative_moe')
            env = dict(os.environ, EXL3_OVERLAY_HOST=str(overlay), CACHE_ROOT=str(cache))
            def resolve():
                return subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True)
            self.assertEqual(resolve().stdout.strip(), str(bundle))
            (bundle/'manifest.json').unlink()
            result = resolve()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('complete manifest bundle', result.stderr)
            self.assertEqual(result.stdout, '')
            (bundle/'runtime.py').unlink()
            self.assertNotEqual(resolve().returncode, 0)
            overlay.write_text('# legacy adapter\n_coop_setup["install"]()\n')
            self.assertEqual(resolve().stdout.strip(), str(cache/'cooperative_moe'))

    def test_rank_wiring(self):
        source=(ROOT/'start-tp3.sh').read_text()
        self.assertEqual(source.count('python3 /opt/glm53/patch_flashkda_tp3.py --root'),2)
        self.assertIn('GLM53_KDA_BF16_LARGE_M GLM53_COOP_GEOMETRY HAREM_KDA_FLASHKDA; do',source)
        self.assertIn('-e "HAREM_KDA_FLASHKDA=$HAREM_KDA_FLASHKDA"',source)
        self.assertIn('"$FLASHKDA_PATCH_HOST" "${ssh_t}:/tmp/patch_flashkda_tp3.py"',source)
        self.assertIn('verify-artifacts "$src"',source)
        self.assertIn('stage_bundle.py',source)
        subprocess.run(['bash','-n',str(ROOT/'start-tp3.sh')],check=True)

    def test_tp3_keeps_208_without_182_and_ablit_off(self):
        source = (ROOT / 'start-tp3.sh').read_text()
        self.assertIn('\nABLIT=0\n', source)
        self.assertIn('unset EXL3_OVERLAY_HOST', source)
        self.assertIn('unset GLM53_EXL3_MOE_FAST', source)
        self.assertIn('unset GLM53_KDA_FP8_FAT', source)
        self.assertNotIn('unset GLM53_KDA_BF16_LARGE_M', source)
        self.assertIn('GLM53_KDA_BF16_LARGE_M="${GLM53_KDA_BF16_LARGE_M-0}"', source)
        self.assertIn('GLM53_KDA_BF16_LARGE_M GLM53_COOP_GEOMETRY HAREM_KDA_FLASHKDA', source)
        self.assertIn('-e GLM53_KDA_BF16_LARGE_M="$GLM53_KDA_BF16_LARGE_M"', source)
        self.assertNotIn('GLM53_EXL3_MOE_FAST="${GLM53_EXL3_MOE_FAST-0}"', source)
        self.assertNotIn('-e GLM53_KDA_FP8_FAT=', source)
        self.assertIn('HAREM_KDA_FLASHKDA="${HAREM_KDA_FLASHKDA:-0}"', source)
        example = (ROOT / '.env.tp3.example').read_text()
        self.assertIn('ABLIT=0', example.splitlines())
        self.assertIn('GLM53_APC_RETENTION_INTERVAL_SWA=0', example.splitlines())
        self.assertNotIn('GLM53_KDA_FP8_FAT', example)
        start = (ROOT / 'start.sh').read_text()
        self.assertIn('\nABLIT=0\n', start)
        self.assertNotIn('HAREM_KDA_FLASHKDA', start)
        self.assertNotIn('patch_flashkda_tp3.py', start)

    def test_profile_covers_adaptive_rows(self):
        text=(ROOT/'examples/tp3-throughput.env').read_text()
        graph_line=next(x for x in text.splitlines() if x.startswith('EXTRA_ARGS='))
        rows=list(map(int,graph_line.split('--cudagraph-capture-sizes ')[1].rstrip('"').split()))
        self.assertEqual(rows, sorted({n*s for n in range(1,9) for s in (1,3,5,8)}))
        self.assertIn('EXL3_TEMP_ROWS_FUSED=64',text)

    def test_policy_complete_and_identity_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs=[]
            for rank in range(3):
                for geom in range(3):
                    records=[{'stage':'profile_identity','native_sha256':'fixture'}]
                    for rows in select_policy.ROWS:
                        for pattern in select_policy.PATTERNS:
                            records.extend([{'stage':'compare','pass':True,'label':f'profile/rank={rank}/geometry={geom}/rows={rows}/{pattern}'},
                                {'stage':'profile','rank':rank,'geometry':geom,'rows':rows,'pattern':pattern,'stock':{'median_ms':1},'candidate':{'median_ms':.9 if geom==1 else 1.1}}])
                    records.append({'stage':'profile_complete','rank':rank,'geometry':geom})
                    path=Path(tmp)/f'{rank}-{geom}.jsonl';path.write_text('\n'.join(json.dumps(x) for x in records));logs.append(path)
            self.assertEqual(select_policy.select(logs,'fixture')['rows']['64'],1)
            for files,digest in ((logs[:-1],'fixture'),(logs,'wrong'),(logs+logs[:1],'fixture')):
                with self.assertRaises(ValueError):select_policy.select(files,digest)
            logs[0].write_text(logs[0].read_text().replace('"pass": true','"pass": false',1))
            with self.assertRaises(ValueError):select_policy.select(logs,'fixture')


if __name__=='__main__':unittest.main()
