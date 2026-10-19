"""Verify SHA-256 checksums of distributed files and decompressed data."""
from pathlib import Path
import gzip, hashlib, json

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text())
for item in manifest['files']:
    p=root/item['path']
    assert p.is_file(),item['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
data=json.loads((root/'data/DATA_MANIFEST.json').read_text())
for item in data['compressed_files']:
    content=gzip.decompress((root/item['path']).read_bytes())
    assert len(content)==item['uncompressed_bytes'],item['path']
    assert hashlib.sha256(content).hexdigest()==item['uncompressed_sha256'],item['path']
print('Verified',len(manifest['files']),'release files and',len(data['compressed_files']),'compressed data files.')
