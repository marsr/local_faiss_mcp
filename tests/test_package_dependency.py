"""Check the dependency contract shipped to clean package installers."""
from email.parser import BytesParser
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from packaging.requirements import Requirement


def test_built_wheel_requires_compatible_mcp(tmp_path):
    root = Path(__file__).resolve().parents[1]
    # Build a copy, keeping build products out of the checkout.
    source = tmp_path / 'source'
    source.mkdir()
    for name in ['pyproject.toml', 'README.md', 'LICENSE', 'MANIFEST.in']:
        shutil.copy(root / name, source / name)
    shutil.copytree(root / 'local_faiss_mcp', source / 'local_faiss_mcp',
                    ignore=shutil.ignore_patterns('__pycache__'))
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                    '--no-build-isolation', '--wheel-dir', str(tmp_path), str(source)],
                   check=True, capture_output=True, text=True)
    with zipfile.ZipFile(next(tmp_path.glob('*.whl'))) as wheel:
        metadata = BytesParser().parsebytes(wheel.read(next(
            name for name in wheel.namelist() if name.endswith('.dist-info/METADATA'))))
    requirement = next(Requirement(value) for value in metadata.get_all('Requires-Dist')
                       if Requirement(value).name == 'mcp')
    assert requirement.specifier.contains('1.30.0')
    assert not requirement.specifier.contains('2.0.0')
    assert not requirement.specifier.contains('3.0.0')
    assert str(requirement) == str(Requirement(next(
        line for line in (root / 'requirements.txt').read_text().splitlines()
        if line.startswith('mcp'))))

    # Exercise pip's clean resolver offline with both SDK major versions offered.
    # Only the dependency contract is under test; these are metadata-only wheels.
    import json
    for version in ['1.30.0', '2.0.0']:
        dist_info = f'mcp-{version}.dist-info'
        with zipfile.ZipFile(tmp_path / f'mcp-{version}-py3-none-any.whl', 'w') as candidate:
            candidate.writestr(f'{dist_info}/METADATA',
                              f'Metadata-Version: 2.1\nName: mcp\nVersion: {version}\n')
            candidate.writestr(f'{dist_info}/WHEEL',
                              'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
            candidate.writestr(f'{dist_info}/RECORD', '')
    report = tmp_path / 'resolution.json'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run',
                    '--ignore-installed', '--no-index', '--find-links', str(tmp_path),
                    '--report', str(report), str(requirement)],
                   check=True, capture_output=True, text=True)
    selected = json.loads(report.read_text())['install']
    assert [(item['metadata']['name'], item['metadata']['version'])
            for item in selected] == [('mcp', '1.30.0')]


def test_installed_sdk_supports_existing_server_api():
    from mcp.server import Server
    app = Server('compatibility-test', instructions='Test server')
    assert callable(app.list_tools())
    assert callable(app.call_tool())
