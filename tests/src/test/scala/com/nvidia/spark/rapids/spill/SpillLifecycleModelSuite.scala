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

import scala.collection.mutable

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import org.scalatest.funsuite.AnyFunSuite

private[spill] object SpillLifecycleModel {
  sealed trait Phase
  case object Ready extends Phase
  case object Copying extends Phase
  case object Copied extends Phase
  case object Finished extends Phase

  sealed trait Action
  case object Acquire extends Action
  case object Release extends Action
  case object QueueGpuRead extends Action
  case object CompleteGpuRead extends Action
  case object CheckSpillable extends Action
  case object StartSpill extends Action
  case object CompleteCopy extends Action
  case object FinishSpill extends Action
  case object Close extends Action

  val actions: Seq[Action] = Seq(Acquire, Release, QueueGpuRead, CompleteGpuRead,
    CheckSpillable, StartSpill, CompleteCopy, FinishSpill, Close)

  case class State(
      phase: Phase = Ready,
      closed: Boolean = false,
      borrowed: Boolean = false,
      borrowedFromHost: Boolean = false,
      gpuReadPending: Boolean = false,
      gpuReadQueued: Boolean = false,
      spillPrechecked: Boolean = false,
      handleDeviceRef: Boolean = true,
      borrowerDeviceRef: Boolean = false,
      spillDeviceRef: Boolean = false,
      hostOwned: Boolean = false,
      handleReleases: Int = 0) {
    def deviceAlive: Boolean = handleDeviceRef || borrowerDeviceRef || spillDeviceRef
  }

  case class Node(state: State, trace: Vector[Action])

  def next(state: State, action: Action): Option[State] = action match {
    case Acquire if !state.closed && !state.borrowed &&
        (state.hostOwned || state.handleDeviceRef) =>
      Some(state.copy(borrowed = true, borrowedFromHost = state.hostOwned,
        borrowerDeviceRef = !state.hostOwned))
    case Release if state.borrowed =>
      Some(state.copy(borrowed = false, borrowedFromHost = false, borrowerDeviceRef = false))
    case QueueGpuRead if state.borrowed && state.borrowerDeviceRef &&
        (!state.closed || state.phase != Ready) &&
        !state.gpuReadQueued && state.phase != Finished =>
      Some(state.copy(borrowed = false, borrowerDeviceRef = false,
        gpuReadPending = true, gpuReadQueued = true))
    case CompleteGpuRead if state.gpuReadPending =>
      Some(state.copy(gpuReadPending = false))
    case CheckSpillable if state.phase == Ready && !state.closed && !state.borrowed &&
        !state.spillPrechecked && state.handleDeviceRef =>
      Some(state.copy(spillPrechecked = true))
    case StartSpill if state.phase == Ready && !state.closed && state.spillPrechecked &&
        state.handleDeviceRef =>
      Some(state.copy(phase = Copying, spillDeviceRef = true))
    case CompleteCopy if state.phase == Copying =>
      Some(state.copy(phase = Copied, hostOwned = !state.closed, spillDeviceRef = false))
    case FinishSpill if state.phase == Copied && !state.gpuReadPending =>
      Some(state.copy(phase = Finished, handleDeviceRef = false,
        handleReleases = state.handleReleases + (if (state.handleDeviceRef) 1 else 0)))
    case Close if !state.closed && (state.phase != Ready || !state.gpuReadPending) =>
      val releaseNow = state.phase == Ready || state.phase == Finished
      Some(state.copy(closed = true, hostOwned = false,
        handleDeviceRef = if (releaseNow) false else state.handleDeviceRef,
        handleReleases = state.handleReleases +
          (if (releaseNow && state.handleDeviceRef) 1 else 0)))
    case _ => None
  }

  def prematureRelease(state: State, action: Action): Option[State] = {
    next(state, action).map { successor =>
      if ((action == Close && state.phase == Copied) ||
          (action == CompleteCopy && state.phase == Copying && state.closed)) {
        successor.copy(handleDeviceRef = false,
          handleReleases = successor.handleReleases + (if (state.handleDeviceRef) 1 else 0))
      } else {
        successor
      }
    }
  }

  def duplicateRelease(state: State, action: Action): Option[State] = {
    if (action == Close && state.closed && state.handleReleases == 1) {
      Some(state.copy(handleReleases = state.handleReleases + 1))
    } else {
      next(state, action)
    }
  }

  def violations(state: State): Seq[String] = {
    val failures = mutable.ArrayBuffer.empty[String]
    if (state.handleReleases > 1) {
      failures += "device handle reference released more than once"
    }
    if (state.gpuReadPending && !state.deviceAlive) {
      failures += "device allocation released before GPU reader completed"
    }
    if ((state.phase == Copying || state.phase == Copied) && !state.handleDeviceRef) {
      failures += "device handle reference released before spill synchronization"
    }
    if (!state.closed && state.phase == Finished && !state.hostOwned) {
      failures += "successful spill has no recoverable copy"
    }
    if (state.closed && state.phase == Finished &&
        (state.handleDeviceRef || state.spillDeviceRef || state.hostOwned)) {
      failures += "closed handle retains an owned resource"
    }
    failures.toSeq
  }

  def explore(transition: (State, Action) => Option[State] = next): Vector[Node] = {
    val initial = Node(State(), Vector.empty)
    val queue = mutable.Queue(initial)
    val seen = mutable.HashSet(initial.state)
    val result = mutable.ArrayBuffer.empty[Node]
    while (queue.nonEmpty) {
      val node = queue.dequeue()
      result += node
      actions.foreach { action =>
        transition(node.state, action).foreach { successor =>
          if (seen.add(successor)) {
            queue.enqueue(Node(successor, node.trace :+ action))
          }
        }
      }
    }
    result.toVector
  }
}

class SpillLifecycleModelSuite extends AnyFunSuite {
  import SpillLifecycleModel._

  test("bounded device spill lifecycle satisfies safety invariants") {
    val states = explore()
    assert(states.nonEmpty)
    val failure = states.iterator.flatMap(node =>
      violations(node.state).map(_ -> node.trace)).take(1).toList.headOption
    assert(failure.isEmpty, failure.toString)
    assert(states.exists(_.state.phase == Finished))
    assert(states.exists(node => node.state.closed && node.state.phase == Copying))
    assert(states.exists(node => node.state.gpuReadPending && node.state.phase == Copied))
    assert(states.exists(node => node.state.borrowed && node.state.phase == Copying))
  }

  test("borrower reference keeps the allocation alive after handle close") {
    val acquired = next(State(), Acquire).get
    val closed = next(acquired, Close).get
    assert(!closed.handleDeviceRef)
    assert(closed.borrowerDeviceRef)
    assert(closed.deviceAlive)
    assert(violations(closed).isEmpty)
    assert(!next(closed, Release).get.deviceAlive)
  }

  test("materialization can repeat and represent host-backed rematerialization") {
    val acquired = next(State(), Acquire).get
    val released = next(acquired, Release).get
    assert(next(released, Acquire).exists(_.borrowerDeviceRef))

    val finished = explore().find(node =>
      node.state.phase == Finished && !node.state.closed).get.state
    val hostBorrow = next(finished, Acquire).get
    assert(hostBorrow.borrowed && hostBorrow.borrowedFromHost)
    assert(!hostBorrow.borrowerDeviceRef)
    assert(next(hostBorrow, QueueGpuRead).isEmpty)
  }

  test("spill precheck can race with materialization") {
    val prechecked = next(State(), CheckSpillable).get
    val borrowed = next(prechecked, Acquire).get
    assert(next(State(), StartSpill).isEmpty)
    assert(next(next(State(), Acquire).get, CheckSpillable).isEmpty)
    assert(next(borrowed, StartSpill).exists(_.spillDeviceRef))
  }

  test("a live borrower can queue a read after handle close during spill") {
    val started = next(next(State(), CheckSpillable).get, StartSpill).get
    val borrowed = next(started, Acquire).get
    val closed = next(borrowed, Close).get
    val queued = next(closed, QueueGpuRead).get
    assert(queued.gpuReadPending && !queued.borrowerDeviceRef)
    assert(violations(next(queued, CompleteCopy).get).isEmpty)
    val releasedBeforeSync = prematureRelease(queued, CompleteCopy).get
    assert(violations(releasedBeforeSync).contains(
      "device allocation released before GPU reader completed"))
    val readyClosed = next(next(State(), Acquire).get, Close).get
    assert(next(readyClosed, QueueGpuRead).isEmpty)
  }

  test("premature release yields shortest ownership and physical-lifetime traces") {
    val nodes = explore(prematureRelease)
    val ownershipFailure = nodes.find(node => violations(node.state).contains(
      "device handle reference released before spill synchronization")).get
    assert(ownershipFailure.trace.last == Close || ownershipFailure.trace.last == CompleteCopy)
    val lifetimeFailure = nodes.find(node => violations(node.state).contains(
      "device allocation released before GPU reader completed")).get
    assert(lifetimeFailure.trace.contains(QueueGpuRead))
    assert(!lifetimeFailure.state.deviceAlive)
    assert(lifetimeFailure.state.gpuReadPending)
  }

  test("an injected duplicate release falsifies the at-most-once property") {
    val failure = explore(duplicateRelease).find(node => violations(node.state).contains(
      "device handle reference released more than once")).get
    assert(failure.trace == Vector(Close, Close))
    assert(failure.state.handleReleases == 2)
  }
}

class SpillLifecycleConformanceSuite extends SpillUnitTestBase {
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
    val beforeSynchronization = new CountDownLatch(1)
    val resumeSynchronization = new CountDownLatch(1)

    override def postSpill(plan: SpillPlan): Unit = {
      beforeSynchronization.countDown()
      await(resumeSynchronization)
      super.postSpill(plan)
    }
  }

  private def createHandle(): SpillableDeviceBufferHandle = {
    closeOnExcept(DeviceMemoryBuffer.allocate(expectedBytes.length)) { deviceBuffer =>
      withResource(HostMemoryBuffer.allocate(expectedBytes.length)) { hostBuffer =>
        hostBuffer.setBytes(0, expectedBytes, 0, expectedBytes.length)
        deviceBuffer.copyFromHostBuffer(hostBuffer)
      }
      SpillableDeviceBufferHandle(deviceBuffer)
    }
  }

  test("materialized device reference prevents a spill") {
    val deviceStore = new SpillableDeviceStore
    SpillFramework.stores.deviceStore = deviceStore
    withResource(createHandle()) { handle =>
      withResource(handle.materialize()) { borrowed =>
        assert(!handle.spillable)
        assertResult(0L)(deviceStore.spill(handle.approxSizeInBytes))
        assert(borrowed.getRefCount > 1)
      }
      assert(handle.spillable)
      assertResult(handle.approxSizeInBytes)(deviceStore.spill(handle.approxSizeInBytes))
      assert(handle.host.isDefined)
    }
  }

  test("current close-before-queued-read path matches the premature-release transition") {
    val hostStore = new PausingHostStore
    val deviceStore = new PausingDeviceStore
    SpillFramework.stores.hostStore = hostStore
    SpillFramework.stores.deviceStore = deviceStore
    withResource(createHandle()) { handle =>
      withResource(new Cuda.Stream(true)) { readStream =>
        withResource(HostMemoryBuffer.allocate(expectedBytes.length, true)) { readDestination =>
          val executor = Executors.newSingleThreadExecutor()
          try {
            val spilled = executor.submit(new Callable[Long] {
              override def call(): Long = deviceStore.spill(handle.approxSizeInBytes)
            })
            await(hostStore.copyFinished)
            withResource(handle.materialize()) { guard =>
              try {
                withResource(handle.materialize()) { borrowed =>
                  hostStore.resumeCopy.countDown()
                  await(deviceStore.beforeSynchronization)
                  val referencesBeforeClose = guard.getRefCount
                  handle.close()
                  assert(handle.dev.isEmpty)
                  assertResult(referencesBeforeClose - 1)(guard.getRefCount)
                  readDestination.copyFromDeviceBufferAsync(borrowed, readStream)
                }
                deviceStore.resumeSynchronization.countDown()
                assertResult(handle.approxSizeInBytes)(
                  spilled.get(timeoutSeconds, TimeUnit.SECONDS))
                readStream.sync()
                val actualBytes = new Array[Byte](expectedBytes.length)
                readDestination.asByteBuffer.get(actualBytes)
                assert(actualBytes.sameElements(expectedBytes))
              } finally {
                readStream.sync()
              }
            }
            assertResult(0)(deviceStore.numHandles)
            assertResult(0)(hostStore.numHandles)
          } finally {
            hostStore.resumeCopy.countDown()
            deviceStore.resumeSynchronization.countDown()
            executor.shutdown()
            if (!executor.awaitTermination(timeoutSeconds, TimeUnit.SECONDS)) {
              executor.shutdownNow()
            }
            readStream.sync()
          }
        }
      }
    }
  }
}
