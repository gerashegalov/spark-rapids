<!--
Copyright (c) 2026, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Spark installation dependency generator

`gen-spark-home-deps.py` is a checked-in copy of
`NVIDIA/spark-rapids-common/spark-home-deps/gen-spark-home-deps.py`
at commit `3aa14436cea85312ba8b67953a816b6f14d26279`.

The source file SHA-256 is `6a4c96b522a564d9526c3315b5c6605f8ca8ba1bd15019be5e65de5b44640932`.
Run `sha256sum -c build/gen-spark-home-deps.sha256` to verify the copy.
Updating it requires a new common commit and an explicit consumer change.
Maven reads this local file; it does not fetch common sources during a build.
