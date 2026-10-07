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

import ai.rapids.cudf.{ColumnVector, Cuda, DeviceMemoryBuffer, Scalar}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuBloomFilter
import com.nvidia.spark.rapids.jni.BloomFilter

class SpillBloomFilterSyncInvestigationSuite extends SpillUnitTestBase {
  private val timeoutSeconds = 90L
  private val probeRows = 8 * 1024 * 1024
  private val probeHashes = 512
  private val filterBits = 4L * 1024 * 1024

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
    val hostStore = new PausingHostStore
    val deviceStore = new PausingDeviceStore
    SpillFramework.stores.hostStore = hostStore
    SpillFramework.stores.deviceStore = deviceStore

    val filterBuffer = createFilterBuffer()
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
                  closeOnExcept(bloomFilter.mightContainLong(input)) { result =>
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
                filterBuffer.incRefCount()
                guardHeld = true
                hostStore.resumeCopy.countDown()
                await(deviceStore.beforeSynchronization)
                val pendingBeforeClose = !completion.hasCompleted
                val referencesBeforeClose = filterBuffer.getRefCount
                assertResult(2)(referencesBeforeClose)
                bloomFilter.close()
                val pendingAfterClose = !completion.hasCompleted
                assertResult(1)(filterBuffer.getRefCount)
                info(s"pending after return=$pendingAfterReturn, before close=" +
                  s"$pendingBeforeClose, after close=$pendingAfterClose")
                deviceStore.resumeSynchronization.countDown()
                assertResult(filterBuffer.getLength)(spilled.get(timeoutSeconds, TimeUnit.SECONDS))
                assert(pendingAfterReturn && pendingBeforeClose && pendingAfterClose)
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
