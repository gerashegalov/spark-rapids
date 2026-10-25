# Spark installation dependency generator

`gen-spark-home-deps.py` is a checked-in copy of
`NVIDIA/spark-rapids-common/spark-home-deps/gen-spark-home-deps.py`
at commit `c9ba17fec80aee58b7239de6b44b288f789e7d77`.

The source file SHA-256 is `0fbb609b9e3e0444e9cadd14c79992ad8fac05faa01a25f17b17d165d512eae9`.
Run `sha256sum -c build/gen-spark-home-deps.sha256` to verify the copy.
Updating it requires a new common commit and an explicit consumer change.
Maven reads this local file; it does not fetch common sources during a build.
