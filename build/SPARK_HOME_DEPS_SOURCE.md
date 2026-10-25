# Spark installation dependency generator

`gen-spark-home-deps.py` is a checked-in copy of
`NVIDIA/spark-rapids-common/spark-home-deps/gen-spark-home-deps.py`
at commit `3aa14436cea85312ba8b67953a816b6f14d26279`.

The source file SHA-256 is `6a4c96b522a564d9526c3315b5c6605f8ca8ba1bd15019be5e65de5b44640932`.
Run `sha256sum -c build/gen-spark-home-deps.sha256` to verify the copy.
Updating it requires a new common commit and an explicit consumer change.
Maven reads this local file; it does not fetch common sources during a build.
