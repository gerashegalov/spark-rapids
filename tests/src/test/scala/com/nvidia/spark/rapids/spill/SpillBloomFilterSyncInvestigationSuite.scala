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

package com.nvidia.spark.rapids.spill

import java.util.concurrent.{Callable, CountDownLatch, Executors, TimeUnit}

import ai.rapids.cudf.{ColumnVector, Cuda, DeviceMemoryBuffer, NvtxColor, NvtxRange, Rmm,
  RmmAllocationMode, Scalar}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuBloomFilter
import com.nvidia.spark.rapids.jni.BloomFilter

class SpillBloomFilterSyncInvestigationSuite extends SpillUnitTestBase {
  private val timeoutSeconds = 90L
  private val probeRows = 8 * 1024 * 1024
  private val probeHashes = 512
  private val filterBits = 4L * 1024 * 1024

  override def afterEach(): Unit = {
    try {
      super.afterEach()
    } finally {
      if (java.lang.Boolean.getBoolean("ctl.spill.rmmAsync") && Rmm.isInitialized) {
        Rmm.shutdown()
      }
    }
  }

  private def await(latch: CountDownLatch): Unit = {
    assert(latch.await(timeoutSeconds, TimeUnit.SECONDS))
  }

  private class PausingHostStore extends SpillableHostStore(Some(64L * 1024 * 1024)) {
    val copyFinished = new CountDownLatch(1)
    val resumeCopy = new CountDownLatch(1)

    override def makeBuilder(
        handle: SpillableHostBufferHandle): SpillableHostBufferHandleBuilder = {
      val underlying = super.makeBuilder(handle)
      new SpillableHostBufferHandleBuilder {
        override def copyNext(buffer: DeviceMemoryBuffer, length: Long,
            stream: Cuda.Stream): Unit = {
          underlying.copyNext(buffer, length, stream)
          copyFinished.countDown()
          await(resumeCopy)
        }

        override def build: SpillableHostBufferHandle = underlying.build

        override def close(): Unit = underlying.close()
      }
    }
  }

  private class PausingDeviceStore extends SpillableDeviceStore {
    val beforeSynchronization = new CountDownLatch(1)
    val resumeSynchronization = new CountDownLatch(1)

    override def postSpill(plan: SpillPlan): Unit = {
      beforeSynchronization.countDown()
      await(resumeSynchronization)
      super.postSpill(plan)
    }
  }

  private def createFilterBuffer(): DeviceMemoryBuffer = {
    withResource(BloomFilter.create(BloomFilter.VERSION_2, probeHashes, filterBits, 0)) { filter =>
      withResource(filter.getListAsColumnView) { view =>
        val source = view.getData
        closeOnExcept(DeviceMemoryBuffer.allocate(source.getLength)) { buffer =>
          buffer.copyFromMemoryBuffer(0, source, 0, source.getLength, Cuda.DEFAULT_STREAM)
          Cuda.memset(buffer.getAddress + 16L, 0xff.toByte, source.getLength - 16L)
          buffer
        }
      }
    }
  }

  test("Bloom probe completion marker remains pending across concurrent spill close") {
    assert(Cuda.isPtdsEnabled())
    val unguarded = java.lang.Boolean.getBoolean("ctl.spill.unguarded")
    val asyncPool = java.lang.Boolean.getBoolean("ctl.spill.rmmAsync")
    val attemptReuse = java.lang.Boolean.getBoolean("ctl.spill.reuse")
    val overwriteReused = java.lang.Boolean.getBoolean("ctl.spill.overwrite")
    assert(!attemptReuse || (unguarded && asyncPool))
    assert(!overwriteReused || attemptReuse)
    if (asyncPool) {
      Rmm.initialize(RmmAllocationMode.CUDA_ASYNC, null, 512L * 1024 * 1024)
    }
    val resource = Option(Rmm.getCurrentDeviceResource).map(_.getClass.getName).getOrElse("none")
    info(s"RMM resource=$resource, async=$asyncPool, unguarded=$unguarded")
    val hostStore = new PausingHostStore
    val deviceStore = new PausingDeviceStore
    SpillFramework.stores.hostStore = hostStore
    SpillFramework.stores.deviceStore = deviceStore

    val filterBuffer = createFilterBuffer()
    val filterAddress = filterBuffer.getAddress
    val bloomFilter = closeOnExcept(filterBuffer)(new GpuBloomFilter(_))
    withResource(bloomFilter) { _ =>
      withResource(Scalar.fromLong(42L)) { value =>
        withResource(ColumnVector.fromScalar(value, probeRows)) { input =>
          withResource(new Cuda.Event()) { completion =>
            Cuda.deviceSynchronize()
            val probeExecutor = Executors.newSingleThreadExecutor()
            val spillExecutor = Executors.newSingleThreadExecutor()
            var guardHeld = false
            try {
              val probe = probeExecutor.submit(new Callable[ColumnVector] {
                override def call(): ColumnVector = {
                  val probed = withResource(new NvtxRange("CtlBloomProbeCall", NvtxColor.BLUE)) {
                    _ => bloomFilter.mightContainLong(input)
                  }
                  closeOnExcept(probed) { result =>
                    completion.record()
                    result
                  }
                }
              })
              withResource(probe.get(timeoutSeconds, TimeUnit.SECONDS)) { result =>
                assertResult(probeRows)(result.getRowCount)
                val pendingAfterReturn = !completion.hasCompleted
                val spilled = spillExecutor.submit(new Callable[Long] {
                  override def call(): Long = deviceStore.spill(filterBuffer.getLength)
                })
                await(hostStore.copyFinished)
                if (!unguarded) {
                  filterBuffer.incRefCount()
                  guardHeld = true
                }
                hostStore.resumeCopy.countDown()
                await(deviceStore.beforeSynchronization)
                val pendingBeforeClose = !completion.hasCompleted
                val referencesBeforeClose = filterBuffer.getRefCount
                assertResult(if (unguarded) 1 else 2)(referencesBeforeClose)
                // A production overlap could involve one task completing and closing its
                // Bloom expression while another task's allocation failure spills the filter.
                // This forced close tests the ordering, not whether a real task completion
                // leaves the Bloom probe running.
                val closeStarted = System.nanoTime()
                withResource(new NvtxRange("CtlBloomHandleClose", NvtxColor.RED)) { _ =>
                  bloomFilter.close()
                }
                val closeElapsedNanos = System.nanoTime() - closeStarted
                val pendingAfterClose = !completion.hasCompleted
                assertResult(if (unguarded) 0 else 1)(filterBuffer.getRefCount)
                info(s"pending after return=$pendingAfterReturn, before close=" +
                  s"$pendingBeforeClose, after close=$pendingAfterClose, " +
                  s"close elapsed ns=$closeElapsedNanos")
                def finishSpill(): Unit = {
                  deviceStore.resumeSynchronization.countDown()
                  val spilledBytes = spilled.get(timeoutSeconds, TimeUnit.SECONDS)
                  assertResult(filterBuffer.getLength)(spilledBytes)
                  assert(pendingAfterReturn && pendingBeforeClose &&
                    (unguarded || pendingAfterClose))
                }
                if (attemptReuse) {
                  withResource(DeviceMemoryBuffer.allocate(filterBuffer.getLength)) { reused =>
                    info(s"filter address=$filterAddress, replacement address=" +
                      s"${reused.getAddress}, same=${filterAddress == reused.getAddress}")
                    if (overwriteReused) {
                      assertResult(filterAddress)(reused.getAddress)
                      Cuda.memset(reused.getAddress + 16L, 0.toByte, reused.getLength - 16L)
                    }
                    try {
                      finishSpill()
                      if (overwriteReused) {
                        withResource(result.copyToHost()) { hostResult =>
                          val falseTailCount = (probeRows - 4096 until probeRows)
                            .count(index => !hostResult.getBoolean(index))
                          info(s"false results in final 4096 rows=$falseTailCount")
                          assert(falseTailCount > 0)
                        }
                      }
                    } finally {
                      Cuda.deviceSynchronize()
                    }
                  }
                } else {
                  finishSpill()
                }
              }
            } finally {
              hostStore.resumeCopy.countDown()
              deviceStore.resumeSynchronization.countDown()
              Cuda.deviceSynchronize()
              if (guardHeld) {
                filterBuffer.close()
              }
              probeExecutor.shutdown()
              spillExecutor.shutdown()
              if (!probeExecutor.awaitTermination(timeoutSeconds, TimeUnit.SECONDS)) {
                probeExecutor.shutdownNow()
              }
              if (!spillExecutor.awaitTermination(timeoutSeconds, TimeUnit.SECONDS)) {
                spillExecutor.shutdownNow()
              }
            }
          }
        }
      }
    }
    assertResult(0)(deviceStore.numHandles)
    assertResult(0)(hostStore.numHandles)
  }
}
