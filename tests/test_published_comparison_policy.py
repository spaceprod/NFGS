"""Protect published scores and live-worker timing provenance."""
import json,tempfile,unittest
from pathlib import Path
from nfgs.comparison_policy import POLICY_ID,file_sha,published_rows,require_allowed_job,normalize_pause_timings

class PublishedComparisonPolicyTests(unittest.TestCase):
    def test_published_values_are_immutable_and_have_no_local_cci(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'published.csv';path.write_text('case,method,C_vec,sigma_J\nTD1,NFPT,0.935,0.102\n')
            (root/'comparison_policy.json').write_text(json.dumps({'policy_id':POLICY_ID,'published_baseline':{'snapshot':'published.csv','sha256':file_sha(path)}}))
            row=published_rows(root)[0]
            self.assertEqual((row['C_vec'],row['sigma_J']),('0.935','0.102'))
            self.assertIsNone(row['E_rel_Bgt']);self.assertIsNone(row['runtime_s'])
            path.write_text('case,method,C_vec,sigma_J\nTD1,NFPT,0.950,0.079\n')
            with self.assertRaises(ValueError):published_rows(root)
    def test_fixed_published_queue_cannot_execute_local_nfpt(self):
        policy={'policy_id':POLICY_ID,'active_methods':['NFGS','NF2']}
        require_allowed_job({'method':'NF2'},policy)
        for method in ['NFPT_Fortran','NFPT_GPU','NFGS_legacy']:
            with self.assertRaises(ValueError):require_allowed_job({'method':method},policy)
    def test_resume_timing_removes_pause_once_and_preserves_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'runs'/'nf2';out.mkdir(parents=True)
            (out/'pause_intervals.json').write_text(json.dumps([{'duration_s':30.}]))
            result=out/'result.json';result.write_text(json.dumps({'training_s':100.,'solver_only_s':100.,'end_to_end_s':115.,'inference_s':10.}))
            for _ in range(2):normalize_pause_timings(root,{'id':'nf2','method':'NF2'})
            r=json.loads(result.read_text())
            self.assertEqual((r['training_s'],r['solver_only_s'],r['end_to_end_s']),(70.,70.,85.))
            self.assertEqual(r['inference_s'],10.);self.assertEqual(r['pause_timing_adjustment']['raw_wall_times']['training_s'],100.)

    def test_clock_shift_is_removed_from_wall_timer_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'runs'/'nf2';out.mkdir(parents=True)
            (out/'pause_intervals.json').write_text(json.dumps([{'duration_s':30.,'WSL_btime_clock_offset_s':7.}]))
            result=out/'result.json';result.write_text(json.dumps({'training_s':100.,'solver_only_s':100.,'end_to_end_s':122.,'inference_s':10.}))
            normalize_pause_timings(root,{'id':'nf2','method':'NF2'})
            r=json.loads(result.read_text())
            self.assertEqual(r['training_s'],70.);self.assertEqual(r['end_to_end_s'],85.)
    def test_old_pause_is_not_subtracted_from_a_new_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'runs'/'nf2';out.mkdir(parents=True)
            (out/'pause_intervals.json').write_text(json.dumps([{'duration_s':30.,'start':1.,'end':31.}]))
            result=out/'result.json';result.write_text(json.dumps({'training_s':100.,'end_to_end_s':110.,'measurement_started_at_wall':1000.}))
            normalize_pause_timings(root,{'id':'nf2','method':'NF2'})
            r=json.loads(result.read_text());self.assertEqual(r['training_s'],100.);self.assertNotIn('pause_timing_adjustment',r)

if __name__=='__main__':unittest.main()
