/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.fileio.hadoop;

import ai.rapids.cudf.HostMemoryBuffer;
import com.nvidia.spark.rapids.IntRangeWithOffset;
import com.nvidia.spark.rapids.PerfIO$;
import com.nvidia.spark.rapids.RangeWithOffset;
import com.nvidia.spark.rapids.SuffixRangeWithOffset;
import com.nvidia.spark.rapids.jni.fileio.RapidsInputFile;
import com.nvidia.spark.rapids.jni.fileio.SeekableInputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import scala.Option;
import scala.collection.JavaConverters;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;

/**
 * S3-backed {@link RapidsInputFile} for Hadoop-conf-driven (non-iceberg) reads.
 * {@code readVectored} issues batched byte-range GETs through the optimized
 * vectored-read path; the other operations delegate to the standard
 * {@link HadoopInputFile}.
 */
public class S3InputFile implements RapidsInputFile {
    private final HadoopInputFile delegate;
    private final URI fileUri;
    private final Configuration hadoopConf;

    public static S3InputFile create(Path filePath, Configuration conf) throws IOException {
        return new S3InputFile(HadoopInputFile.create(filePath, conf), filePath.toUri(), conf);
    }

    private S3InputFile(HadoopInputFile delegate, URI fileUri, Configuration hadoopConf) {
        this.delegate = delegate;
        this.fileUri = fileUri;
        this.hadoopConf = hadoopConf;
    }

    @Override
    public String path() {
        return delegate.path();
    }

    @Override
    public long getLength() throws IOException {
        return delegate.getLength();
    }

    @Override
    public OptionalLong getLastModificationTime() throws IOException {
        return delegate.getLastModificationTime();
    }

    @Override
    public SeekableInputStream open() throws IOException {
        return delegate.open();
    }

    @Override
    public void readVectored(HostMemoryBuffer output, List<RapidsInputFile.CopyRange> copyRanges)
            throws IOException {
        List<RangeWithOffset> ranges = new ArrayList<>(copyRanges.size());
        for (RapidsInputFile.CopyRange range : copyRanges) {
            ranges.add(new IntRangeWithOffset(
                    range.getInputOffset(), range.getLength(), range.getOutputOffset()));
        }
        readToHostMemory(output, ranges);
    }

    /**
     * Issue a single suffix-range {@code GetObject} ({@code Range: bytes=-N}) for
     * the last {@code length} bytes. Avoids the {@code getLength()} round-trip the
     * default {@link RapidsInputFile#readTail} would make.
     */
    @Override
    public void readTail(long length, HostMemoryBuffer output) throws IOException {
        if (length == 0) {
            return;
        }
        if (length < 0) {
            throw new IllegalArgumentException("length must be non-negative");
        }
        List<RangeWithOffset> ranges = new ArrayList<>(1);
        ranges.add(new SuffixRangeWithOffset(length, 0L));
        readToHostMemory(output, ranges);
    }

    private void readToHostMemory(HostMemoryBuffer output, List<RangeWithOffset> ranges) {
        Option<Object> result = PerfIO$.MODULE$.readToHostMemory(
                hadoopConf,
                output,
                fileUri,
                () -> JavaConverters.asScalaBufferConverter(ranges).asScala().toSeq());
        if (result.isEmpty()) {
            throw new IllegalArgumentException("expected to use PerfIO to read");
        }
    }
}
