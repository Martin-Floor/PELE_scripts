"""Small, dependency-free helpers for PELE output cleanup.

Kept separate from ``_pele_analysis`` so they can be imported and tested without
pulling in BioPython, mdtraj, matplotlib and the rest of the analysis stack.
"""
import hashlib
import os


def _digest(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def topologies_are_identical(topo_dir, files):
    """True only if every topology PDB in ``files`` is byte-identical to the first.

    PELE writes one topology per explorer. In an *induced-fit* run each of those
    files holds a different protein conformation — same atom count, same
    ordering, different coordinates — so deleting all but the first discards the
    per-explorer conformational data. The loss is silent, because trajectories
    still load against any single topology. Deduplication is therefore only safe
    when the files are genuinely redundant.

    The first two files are compared before the rest: distinct conformations
    differ immediately, so the dangerous case is rejected without hashing the
    whole set.

    Parameters
    ----------
    topo_dir : str
        Directory holding the topology PDBs.
    files : list of str
        File names within ``topo_dir``, in a stable order.

    Returns
    -------
    bool
        True when deduplication would not lose information.
    """
    if len(files) < 2:
        return True
    first = _digest(os.path.join(topo_dir, files[0]))
    if _digest(os.path.join(topo_dir, files[1])) != first:
        return False
    return all(_digest(os.path.join(topo_dir, f)) == first for f in files[2:])
