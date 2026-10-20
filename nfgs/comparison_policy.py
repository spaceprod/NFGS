"""Provenance rules for fixed published NFPT and newly executed classical comparisons."""
from pathlib import Path
import csv,hashlib,json

POLICY_ID = "published_nfpt_fixed_v2"
ACTIVE_METHODS = ("NFGS",)

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)

def read_policy(root):
    path=Path(root)/"comparison_policy.json"
    return json.loads(path.read_text()) if path.exists() else None

def classify_job(job):
    return "active_controlled" if job["method"] in ACTIVE_METHODS else "local_implementation_supplement"

def require_allowed_job(job,policy):
    known=set(ACTIVE_METHODS) | {"NFPT_Fortran", "NFPT_GPU", "NFGS_legacy"}
    if job["method"] not in known:raise ValueError("Unsupported comparison method: "+job["method"])
    if policy and policy["policy_id"]==POLICY_ID and job["method"] not in policy["active_methods"]:
        raise ValueError("The fixed-published policy prohibits this solver in the active queue: "+job["method"])

def published_rows(root):
    root=Path(root);policy=read_policy(root)
    if not policy or policy.get("policy_id")!=POLICY_ID:return []
    path=root/policy["published_baseline"]["snapshot"]
    if file_sha(path)!=policy["published_baseline"]["sha256"]:
        raise ValueError("Frozen published scores have changed")
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    # Retain the original decimal strings. Never attach a local CCI or runtime.
    return [{**r,"evidence_origin":"published_original","evaluation_protocol":"yi2022_original",
             "E_rel_Bgt":None,"runtime_s":None,"publication_equivalence_verified":False} for r in rows]

def normalize_pause_timings(root,job):
    """Subtract suspension once, using the timer's clock and retaining raw values."""
    root=Path(root);out=root/"runs"/job["id"];interval_file=out/"pause_intervals.json"
    if not interval_file.exists() or job["method"] not in ACTIVE_METHODS:return
    result_file=out/"result.json";result=json.loads(result_file.read_text())
    previous=result.get("pause_timing_adjustment",{})
    if previous.get("version",0)>=2:return
    keys=("training_s","solver_only_s","end_to_end_s")
    raw=previous.get("raw_wall_times") or {key:result[key] for key in keys if key in result}
    estimated_start=result.get("measurement_started_at_wall",result_file.stat().st_mtime-raw.get("end_to_end_s",0.))
    intervals=[v for v in json.loads(interval_file.read_text()) if "end" not in v or v["end"]>=estimated_start-5.]
    if not intervals:return
    duration=sum(v["duration_s"] for v in intervals)
    clock_shift=sum(v.get("WSL_btime_clock_offset_s",0.) for v in intervals)
    deductions={key:duration for key in raw}
    if result.get("end_to_end_clock","CLOCK_REALTIME")=="CLOCK_REALTIME" and "end_to_end_s" in raw:
        deductions["end_to_end_s"]+=clock_shift
    for key,value in raw.items():result[key]=max(0.,value-deductions[key])
    result["pause_timing_adjustment"]={"version":2,"duration_s":duration,"raw_wall_times":raw,"intervals":intervals,
        "deductions_s":deductions,"end_to_end_clock":result.get("end_to_end_clock","CLOCK_REALTIME"),
        "reason":"same live optimizer retained; steady training clock excludes btime shift, legacy wall end-to-end clock includes it"}
    temporary=result_file.with_suffix(".json.tmp");temporary.write_text(json.dumps(result,indent=2)+"\n");temporary.replace(result_file)
