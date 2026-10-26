# Shared Shimpler source tools

The four Apache-licensed `shimpler_source_*.py` modules in this directory are
vendored unchanged from the `shimpler-source-tools` directory of public
`NVIDIA/spark-rapids-common` PR #64 at commit
`3f93bc37fcabd89c74107a0ba8a57dc343dcd9c0`. This is a provisional pin
until PR #64 merges; update the pin to the merged commit before merging this PR.

The copies are checked in so Maven generation remains offline. Update all four
from one reviewed common commit, update `shimpler_source_tools.sha256`, then
verify `python3 -m unittest discover -s build/tests -p 'test_shimpler*.py'`,
full source-set parity across both Scala POM graphs, and the required clean
multi-shim build before publishing the consumer change.

This repository retains its own Ant adapter, Maven profiles, and build-version
policy. The shared modules contain no repository-specific version matrix.
