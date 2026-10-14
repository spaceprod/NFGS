#!/usr/bin/env python3
"""Reject private solver/build artifacts from publishable paths and Git history."""
from pathlib import Path
import argparse,json,subprocess,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN_EXTENSIONS={'.f','.for','.f90','.f95','.f03','.f08','.mod','.o','.so','.a','.exe'}
FORBIDDEN_ROOTS={'.local','.venv','results','release_data'}


def forbidden(name):
    p=Path(name)
    return p.suffix.lower() in FORBIDDEN_EXTENSIONS or (p.parts and p.parts[0] in FORBIDDEN_ROOTS) or 'build_nfpt_external' in p.name


def check():
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    failures=[p for p in paths if forbidden(p)]
    for name in paths:
        p=ROOT/name
        if p.is_file():
            with p.open('rb') as f:
                if f.read(4)==b'\x7fELF':failures.append(name)
    commits=subprocess.check_output(['git','rev-list','--all'],cwd=ROOT,text=True).splitlines()
    for commit in commits:
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True).splitlines()
        failures.extend(f'{commit}:{p}' for p in names if forbidden(p))
    archive=ROOT/'release_data/nfpt_comparison_fields.tar'
    members=None
    if archive.exists():
        with tarfile.open(archive) as tar:
            members=tar.getmembers()
            expected={f'td{i}__nfpt_fortran_symmetric.npz' for i in (1,2,3)}
            if {m.name for m in members}!=expected or not all(m.isfile() for m in members):
                failures.append('comparison archive contains unexpected solver/build files')
            for member in members:
                if member.isfile():
                    with zipfile.ZipFile(tar.extractfile(member)) as fields:
                        if set(fields.namelist()) != {k+'.npy' for k in ['Bx','By','Bz','Jx','Jy','Jz']}:
                            failures.append('comparison NPZ contains unexpected embedded files: '+member.name)
    if failures:raise SystemExit('Publication blocked: '+json.dumps(failures))
    return {'status':'passed','publishable_files':len(paths),'git_commits_checked':len(commits),
            'private_Fortran_solver_or_binary_in_publishable_tree':False,
            'NFPT_release_members':[m.name for m in members] if members else None}


if __name__=='__main__':print(json.dumps(check(),indent=2))
