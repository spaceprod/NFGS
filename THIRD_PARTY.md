# Third-party attribution

`third_party/FastQSL` contains the FastQSL implementation associated with Zhang
et al. (2022), *FastQSL: A Fast Computation Method for Quasi-Separatrix Layers*.
Its license is Creative Commons Attribution-NonCommercial-ShareAlike 4.0,
retained in `third_party/FastQSL/LICENSE`. This is separate from the NFGS license.
The original archive source and local CUDA kernel adaptations are described in
`third_party/FastQSL/ARCHIVE_SOURCE.md` and its README. Those changes are retained
for the same tracing behavior as the manuscript.

The PT/NFPT formulation follows Yi et al. (2022). The standalone solver is
assembled from the research project's Python implementation; original file
hashes are recorded in `provenance/source_manifest.json` and in the standalone
file. The Titov--Demoulin generator follows the TD99 analytic construction and
the implementation conventions documented in its source.

Frozen NFPT fields are distributed as a separate, checksum-pinned comparison
data archive. They are archived inputs to structural comparisons. The reported
NFPT, VFVP and SSW metric values are quoted from Yi et al. (2022), Tables 2--4;
the archive is not represented as a reimplementation of those published solvers.
