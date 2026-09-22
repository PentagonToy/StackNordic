# Stored-data layout

Direct export supports uncollated `processorN`, collated `processorsN` and grouped collated `processorsN_a-b` layouts. The complete decomposition and global cell permutation are validated independently at every selected physical time.

Time-local `polyMesh/cellProcAddressing` takes precedence over `constant/polyMesh/cellProcAddressing`. Rank-count changes, repartitioning and moving or topology-changing meshes therefore do not inherit stale addressing from an earlier time.

Cartesian coordinates are emitted only when a single-block `blockMeshDict` is verified against the global cell count and no selected time contains a local mesh. All other exports use stable global cell indices and record `layout="cells"`.
