"""Unit tests for grouped splitting and review validation in the task 2 manifest builder."""
import sys
from pathlib import Path
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fma_project.manifest import connected_groups, assign_splits, validate_review, load_decisions
from fma_project.audit import AuditError


class ManifestTests(unittest.TestCase):
    def test_transitive_links_and_names(self):
        data = pd.DataFrame({'artist_id':[1,1,2,3], 'album_id':[10,11,11,12],
                             'artist_name_normalized':['a','a','b','b'],
                             'feature_fingerprint':[None]*4}, index=pd.Index([1,2,3,4],name='track_id'))
        groups, _ = connected_groups(data, pd.DataFrame(columns=['track_id_a','track_id_b','reason']))
        self.assertEqual(groups.nunique(),1)

    def test_split_is_group_disjoint_and_order_independent(self):
        groups = pd.Series([i//3 for i in range(300)], index=range(300))
        original = pd.Series(['training','validation','test']*100,index=groups.index)
        first, method = assign_splits(groups,original)
        second, _ = assign_splits(groups.iloc[::-1],original.iloc[::-1])
        pd.testing.assert_series_equal(first,second)
        self.assertEqual(method,'group_shuffle_80_10_10')
        self.assertEqual(pd.DataFrame({'g':groups,'s':first}).groupby('g')['s'].nunique().max(),1)
        self.assertEqual(set(first),{'training','validation','test'})

    def test_valid_existing_split_is_preserved(self):
        groups = pd.Series(range(30))
        original = pd.Series(['training']*24+['validation']*3+['test']*3)
        split, method = assign_splits(groups,original)
        self.assertEqual(method,'preserved_existing_split')
        self.assertEqual(split.tolist(),original.tolist())

    def test_missing_evidence_cannot_freeze(self):
        with self.assertRaises(AuditError): validate_review(None,'candidate')
        decisions=load_decisions(None,pd.Index([1,2],name='track_id'))
        self.assertTrue(decisions['eligibility'].eq('pending').all())
        self.assertTrue(decisions['audio_status'].eq('pending').all())

if __name__ == '__main__': unittest.main()
