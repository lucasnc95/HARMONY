#!/usr/bin/env python3
"""Capture frontend build provenance for controlled benchmark execution."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def command(*argv):
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root, bin_dir, compiler, library, *flags = sys.argv[1:]
    root, bin_dir = Path(root), Path(bin_dir)
    sources = sorted(set(
        path for directory in (root / 'dcl', root / 'benchmarks')
        for pattern in ('*.hpp', '*.cpp', '*.cl')
        for path in directory.rglob(pattern)
    ))
    source_hash = hashlib.sha256()
    for source in sources:
        source_hash.update(str(source.relative_to(root)).encode() + b'\0' + source.read_bytes())
    revision = command('git', '-C', str(root), 'rev-parse', 'HEAD')
    dirty = command('git', '-C', str(root), 'status', '--porcelain')
    opencl_header = root / 'third_party/opencl_headers/CL/cl.h'
    opencl_file = Path(library)
    data = dict(schema_version=1, source_revision=revision,
                working_tree_dirty=None if dirty is None else bool(dirty),
                source_sha256=source_hash.hexdigest(),
                compiler=compiler, compiler_version=command(compiler, '--version'),
                flags=flags, opencl_library=library,
                opencl_headers=str(root / 'third_party/opencl_headers'),
                opencl_header_sha256=sha256(opencl_header) if opencl_header.is_file() else None,
                opencl_library_sha256=sha256(opencl_file) if opencl_file.is_file() else None,
                commands={}, binaries={})
    for workload in ('kneighbor', 'leanmd'):
        binary = bin_dir / f'{workload}.out'
        argv = [compiler, *flags, str(root / 'benchmarks' / f'{workload}.cpp'),
                library, '-o', str(binary)]
        data['commands'][workload] = argv
        data['binaries'][workload] = dict(path=str(binary), sha256=sha256(binary))
    halo = bin_dir / 'test_async_halo_opencl.out'
    data['binaries']['test_async_halo_opencl'] = dict(
        path=str(halo), sha256=sha256(halo))
    destination = bin_dir / 'build_manifest.json'
    temporary = destination.with_name(destination.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary.replace(destination)
    print(f'Build manifest: {destination}')


if __name__ == '__main__':
    main()
