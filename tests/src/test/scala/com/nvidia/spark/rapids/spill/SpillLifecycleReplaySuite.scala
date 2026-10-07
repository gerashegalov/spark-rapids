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

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer}
import com.nvidia.spark.rapids.Arm.withResource

class SpillLifecycleReplaySuite extends SpillUnitTestBase {
  private val timeoutSeconds = 15L
  private val expectedBytes = Array.tabulate[Byte](64)(index => (index * 3).toByte)

  private def await(latch: CountDownLatch): Unit = {
    assert(latch.await(timeoutSeconds, TimeUnit.SECONDS))
  }

  private class PausingHostStore extends SpillableHostStore(Some(1024L)) {
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
    val beforeRelease = new CountDownLatch(1)
    val resumeRelease = new CountDownLatch(1)

    override def postSpill(plan: SpillPlan): Unit = {
      beforeRelease.countDown()
      await(resumeRelease)
      super.postSpill(plan)
    }
  }

  private def createHandle(): SpillableDeviceBufferHandle = {
    val deviceBuffer = DeviceMemoryBuffer.allocate(expectedBytes.length)
    withResource(HostMemoryBuffer.allocate(expectedBytes.length)) { hostBuffer =>
      hostBuffer.setBytes(0, expectedBytes, 0, expectedBytes.length)
      deviceBuffer.copyFromHostBuffer(hostBuffer)
    }
    SpillableDeviceBufferHandle(deviceBuffer)
  }

  private def assertContents(buffer: DeviceMemoryBuffer): Unit = {
    withResource(HostMemoryBuffer.allocate(expectedBytes.length)) { hostBuffer =>
      hostBuffer.copyFromDeviceBuffer(buffer)
      val actual = new Array[Byte](expectedBytes.length)
      hostBuffer.asByteBuffer.get(actual)
      assert(actual.sameElements(expectedBytes))
    }
  }

  private def runScenario(
      atCopy: SpillableDeviceBufferHandle => Unit,
      atPublication: SpillableDeviceBufferHandle => Unit): Unit = {
    val hostStore = new PausingHostStore
    val deviceStore = new PausingDeviceStore
    SpillFramework.stores.hostStore = hostStore
    SpillFramework.stores.deviceStore = deviceStore
    val executor = Executors.newSingleThreadExecutor()
    val handle = createHandle()
    try {
      val spilled = executor.submit(new Callable[Long] {
        override def call(): Long = deviceStore.spill(handle.approxSizeInBytes)
      })
      await(hostStore.copyFinished)
      atCopy(handle)
      hostStore.resumeCopy.countDown()
      await(deviceStore.beforeRelease)
      atPublication(handle)
      deviceStore.resumeRelease.countDown()
      assertResult(handle.approxSizeInBytes)(spilled.get(timeoutSeconds, TimeUnit.SECONDS))
      handle.close()
      assertResult(0)(deviceStore.numHandles)
      assertResult(0)(hostStore.numHandles)
      assertResult(0)(SpillFramework.stores.diskStore.numHandles)
    } finally {
      hostStore.resumeCopy.countDown()
      deviceStore.resumeRelease.countDown()
      executor.shutdown()
      if (!executor.awaitTermination(timeoutSeconds, TimeUnit.SECONDS)) {
        executor.shutdownNow()
      }
      handle.close()
    }
  }

  test("materialize during host copy and recover after spill") {
    var borrowed: DeviceMemoryBuffer = null
    try {
      runScenario(
        handle => {
          borrowed = handle.materialize()
          assertContents(borrowed)
        },
        handle => {
          assert(handle.host.isDefined)
          assertContents(borrowed)
          borrowed.close()
          borrowed = null
          withResource(handle.materialize())(assertContents)
        })
    } finally {
      if (borrowed != null) {
        borrowed.close()
      }
    }
  }

  test("close during host copy releases all stores") {
    runScenario(
      handle => handle.close(),
      handle => {
        assert(handle.closed)
        assert(handle.dev.isDefined)
      })
  }

  test("close after host copy waits for post-spill synchronization") {
    runScenario(
      _ => (),
      handle => {
        handle.close()
        assert(handle.dev.isDefined)
      })
  }
}
