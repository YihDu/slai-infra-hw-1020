import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from optimizer.compiler import compile_candidate


class CompilerTests(unittest.TestCase):
    def test_timeout_removes_container_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['workload.json','config.json','kernel.py']:
                (root/name).write_text('{}')
            completed=SimpleNamespace(stdout='sha256:test\n',stderr='',returncode=0)
            with patch('optimizer.compiler.subprocess.run',side_effect=[completed,subprocess.TimeoutExpired('docker',1),completed]) as run:
                report=compile_candidate(root/'workload.json',root/'config.json',root/'kernel.py',root/'out',timeout=1)
            self.assertEqual(report['status'],'timeout')
            self.assertEqual(report['container_image_id'],'sha256:test')
            command=run.call_args_list[1].args[0]
            name=command[command.index('--name')+1]
            self.assertEqual(run.call_args_list[2].args[0][-3:],['rm','-f',name])
            self.assertIn('--network=none',command)
            self.assertTrue((root/'out/kernel.py').exists())
            self.assertEqual(json.loads((root/'out/compilation.json').read_text())['status'],'timeout')
            with self.assertRaises(ValueError):
                compile_candidate(root/'workload.json',root/'config.json',root/'kernel.py',root/'out')

    def test_missing_docker_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_text('{}')
            with patch('optimizer.compiler.subprocess.run',side_effect=FileNotFoundError):
                report=compile_candidate(source,source,source,root/'out')
            self.assertEqual(report['status'],'unavailable')


if __name__=='__main__':unittest.main()
