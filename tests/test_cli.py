import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        return subprocess.run([str(ROOT / 'img_to_vox.sh'), *map(str, args)], cwd=cwd,
                              text=True, capture_output=True,
                              env={**os.environ, 'HF_HUB_OFFLINE': '1'}, timeout=30)

    def test_help_outside_project(self):
        result = self.run_cli('--help', cwd='/tmp')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--input_folder', result.stdout)

    def test_skip_does_not_load_models(self):
        with tempfile.TemporaryDirectory(prefix='voxel CLI ') as tmp:
            folder = Path(tmp)
            (folder / 'test.png').write_bytes(b'not read when skipped')
            (folder / 'vox').mkdir()
            (folder / 'vox/test.vox').write_bytes(b'existing output')
            result = self.run_cli('--input_folder', '.', '--skip', cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('skipped=1', result.stdout)
            self.assertNotIn('[runtime]', result.stdout)
            self.assertEqual((folder / 'vox/test.vox').read_bytes(), b'existing output')

    def test_duplicate_stems_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('a.jpg', 'a.png'):
                (Path(tmp) / name).touch()
            result = self.run_cli('--input_folder', tmp)
            self.assertEqual(result.returncode, 2)
            self.assertIn('same stem', result.stderr)

    def test_invalid_height(self):
        result = self.run_cli('-i', 'unused.png', '-h', '0')
        self.assertEqual(result.returncode, 2)
        self.assertIn('max_height', result.stderr)
