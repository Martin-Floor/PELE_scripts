"""Tests for the topology-dedup safety guard in cleanPELEFolder.

Regression cover for a data-loss hazard: `dedup_topologies=True` keeps one
topology PDB and deletes the rest. That is correct only when the files really
are redundant. An induced-fit PELE run writes ONE CONFORMATION PER EXPLORER —
same atom count and ordering, different coordinates — so the deletion is
silent: trajectories still load, but the per-explorer conformational data is
gone. Observed on a 1000-ligand CDK2 campaign, where all 95 topologies per
ligand were distinct.
"""
import importlib.util
import os
import tempfile
import unittest

# Load the helper module directly by path: importing the package would pull in
# BioPython, mdtraj and matplotlib, none of which this pure-function test needs.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    '_clean_utils',
    os.path.join(os.path.dirname(_HERE), 'pele_analysis', '_clean_utils.py'))
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_topologies_are_identical = _MOD.topologies_are_identical


ATOM = ('ATOM  {i:5d}  CA  ALA A{i:4d}    '
        '{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n')


def _write(path, shift=0.0, n=50):
    with open(path, 'w') as fh:
        for i in range(1, n + 1):
            fh.write(ATOM.format(i=i, x=1.0 + shift, y=2.0, z=3.0))
        fh.write('END\n')


class TopologyGuard(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def _files(self):
        return sorted(f for f in os.listdir(self.d)
                      if f.startswith('topology_') and f.endswith('.pdb'))

    def test_identical_topologies_allow_dedup(self):
        """Genuinely redundant files: dedup is safe and must be permitted."""
        for k in range(5):
            _write(os.path.join(self.d, f'topology_{k}.pdb'))
        self.assertTrue(_topologies_are_identical(self.d, self._files()))

    def test_distinct_conformations_block_dedup(self):
        """Induced-fit case: every file a different conformation."""
        for k in range(5):
            _write(os.path.join(self.d, f'topology_{k}.pdb'), shift=0.01 * k)
        self.assertFalse(_topologies_are_identical(self.d, self._files()))

    def test_single_differing_file_late_in_the_set_is_caught(self):
        """The cheap two-file pre-check must not create a false negative:
        a set that only diverges at the last file must still be rejected."""
        for k in range(5):
            _write(os.path.join(self.d, f'topology_{k}.pdb'))
        _write(os.path.join(self.d, 'topology_4.pdb'), shift=0.5)
        self.assertFalse(_topologies_are_identical(self.d, self._files()))

    def test_single_topology_is_trivially_identical(self):
        _write(os.path.join(self.d, 'topology_0.pdb'))
        self.assertTrue(_topologies_are_identical(self.d, self._files()))

    def test_empty_set_does_not_raise(self):
        self.assertTrue(_topologies_are_identical(self.d, []))


if __name__ == '__main__':
    unittest.main(verbosity=2)
