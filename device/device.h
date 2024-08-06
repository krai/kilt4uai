//
// MIT License
//
// Copyright (c) 2024 Krai Ltd
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <semaphore>

#include "uai_untether.h"

#if TESTING
#include <dummy_kilt.h>
#else
#include "config/device_config.h"
#endif

#define SIMD_ARGMAX defined(__amd64__) && defined(ENABLE_ZEN2)
// A queue of T where T can be in one of 3 stages: FREE, HAS_SAMPLES, ON_DEVICE
// It expects each state to have 1 writer and 1 reader MAX
template <typename T>
class MultiStageQueue {
 public:
  MultiStageQueue(std::vector<T>& vec) : _free(0), _has_samples(0), _on_device(0), _vec(vec) {}

  enum class Stage {
    FREE = 1,
    HAS_SAMPLES = 2,
    ON_DEVICE = 3
  };

  bool Full(Stage stage) const {
    return free_space(stage) == 0;
  }

  // PRE: The queue for the stage isn't empty
  T& GetFirst(Stage stage) {
#if TESTING
    assert(!Full(stage));
#endif
    std::atomic<size_t>* ptr = nullptr;
    switch (stage) {
      case Stage::FREE:
        ptr = &_free;
        break;
      case Stage::HAS_SAMPLES:
        ptr = &_has_samples;
        break;
      case Stage::ON_DEVICE:
        ptr = &_on_device;
    }
    return _vec.at((*ptr) % depth());
  }

  void Push(Stage stage) {
    switch (stage) {
      case Stage::FREE:
        _on_device++;
        break;
      case Stage::HAS_SAMPLES:
        _free++;
        break;
      case Stage::ON_DEVICE:
        _has_samples++;
    }
  }

 private:
  inline size_t free_space(Stage stage) const {
    switch (stage) {
      case Stage::FREE:
        return depth() + _on_device - _free;
      case Stage::HAS_SAMPLES:
        return _free - _has_samples;
      case Stage::ON_DEVICE:
        return _has_samples - _on_device;
    }
    assert(false);
  }

  size_t depth() const {
    return _vec.size();
  }

  std::atomic<size_t> _free, _has_samples, _on_device;
  std::vector<T>& _vec;
};

class DeviceBuffers {
 public:
  DeviceBuffers(UaiModule* mod, size_t batch_size) {
    size_t num_buffers;
    uai_module_get_num_streams(mod, &num_buffers);
    std::vector<UaiDataStreamInfo> buffer_infos(num_buffers);
    uai_module_get_stream_info(mod, buffer_infos.data(), num_buffers);

    UaiDataBuffer* prev_buffer = nullptr;
    for (auto buffer_info : buffer_infos) {
      auto buffer = std::make_unique<UaiDataBuffer>();

      memset(buffer.get(), 0, sizeof(UaiDataBuffer));
      uai_module_data_buffer_attach(mod, buffer.get(), buffer_info.name, buffer_info.framesize_hint * batch_size);

      if (prev_buffer != nullptr)
        prev_buffer->next_buffer = buffer.get();
      prev_buffer = buffer.get();

      switch (buffer_info.io_type) {
        case UAI_DATA_STREAM_HOST_TO_DEVICE:
          _input_buffers.push_back(std::move(buffer));
          break;
        case UAI_DATA_STREAM_DEVICE_TO_HOST:
          _output_buffers.push_back(std::move(buffer));
          break;
      }
    }

    for (auto& buf : _input_buffers) {
      _raw_input_buffers.push_back(buf->buffer);
    }
    for (auto& buf : _output_buffers) {
      _raw_output_buffers.push_back(buf->buffer);
    }
  }

  DeviceBuffers() = delete;
  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers(DeviceBuffers&&) = default;

  UaiDataBuffer* GetInputBuffer(size_t idx) const {
    return _input_buffers.at(idx).get();
  }

  UaiDataBuffer* GetOutputBuffer(size_t idx) const {
    return _output_buffers.at(idx).get();
  }

  void* GetInputBufferRaw(size_t idx) const {
    return GetInputBuffer(idx)->buffer;
  }

  void* GetOutputBufferRaw(size_t idx) const {
    return GetOutputBuffer(idx)->buffer;
  }

  const std::vector<void*> GetRawInputBuffers() const {
    return _raw_input_buffers;
  }

  const std::vector<void*> GetRawOutputBuffers() const {
    return _raw_output_buffers;
  }

 private:
  std::vector<std::unique_ptr<UaiDataBuffer>> _input_buffers;
  std::vector<std::unique_ptr<UaiDataBuffer>> _output_buffers;

  std::vector<void*> _raw_input_buffers;
  std::vector<void*> _raw_output_buffers;
};

template <typename Sample>
struct Payload {
  Payload(UaiModule* mod, size_t batch_size) : _buffers(mod, batch_size), ignore(false) {}

  const DeviceBuffers& GetBuffers() const {
    return _buffers;
  }

  void Enqueue(UaiModule* mod) {
    memset(&_event, 0, sizeof(UaiEvent));
    _event.buffers = _buffers.GetInputBuffer(0);
    uai_module_enqueue(mod, &_event);
  }

  void Wait(UaiModule* mod, int timeout) {
    while (uai_module_wait(mod, &_event, timeout) != UAI_SUCCESS) {
    };
  }

  std::vector<Sample> samples;
  Payload() = delete;
  Payload(const Payload&) = delete;
  Payload(Payload&&) = default;

 private:
  DeviceBuffers _buffers;
  UaiEvent _event;

 public:
  bool ignore;
};

template <typename Sample>
class Device : public IDevice<Sample> {
  using State = typename IDevice<Sample>::State;

 public:
  Device() : _module(nullptr),
             _payloads_queue(_payloads),
             _state(State::WAITING) {}
  ~Device() {
    _terminate = true;
    for (auto& thread : _threads) {
      thread.join();
    }

    uai_module_free(_module);
  }

  // This is separate from the Constructor to allow us to set thread affinities and setup devices in parallel
  void Construct(IModel* model, IDataSource* data_source, UAIDeviceConfig* device_cfg, int hw_id, std::vector<int> affinities) {
    _device_id = hw_id;
    TRACE_EVENT(device_construct);

    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);

    for (int aff : affinities) {
      CPU_SET(aff, &cpu_set);
    }

    std::binary_semaphore wait_sema{0};

    std::thread load_uai(&Device<Sample>::LoadUAI, this, device_cfg, std::ref(wait_sema));
    pthread_setaffinity_np(load_uai.native_handle(), sizeof(cpu_set_t), &cpu_set);
    wait_sema.release(); // Signal the thread that it can now run
    load_uai.join();

    _scheduler_yield_time = device_cfg->getUaiSchedulerYieldTime();
    _postprocessor_yield_time = device_cfg->getUaiPostprocessorYieldTime();
    _flush_immediately = device_cfg->getUaiFlushImmediately();
    _flush_yield_time = device_cfg->getUaiFlushYieldTime();
    _device_reset_sample_count = device_cfg->getUaiModelBatchSize();
    _before_flush_wait_time = device_cfg->getBeforeFlushWaitTime();
    _wait_timeout = device_cfg->getUaiWaitTimeout();

#if !(SIMD_ARGMAX)
    _buffer.reserve(_batch_size * 1024);
#endif

    _model = model;
    _data_source = data_source;

    std::thread scheduler(&Device<Sample>::QueueScheduler, this);
    std::thread postprocessor(&Device<Sample>::PostProcessor, this);

    _threads.push_back(std::move(scheduler));
    _threads.push_back(std::move(postprocessor));

    for (auto& thread : _threads) {
      pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpu_set);
    }

    _state = State::READY;
  }

  void LoadUAI(UAIDeviceConfig* device_cfg, std::binary_semaphore& wait_sema) {
    wait_sema.acquire(); // Wait until the thread affinity has been set
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Wait a second so that the thread is guaranteed to be moved to the right core

    uai_module_load(device_cfg->getModelRoot().c_str(), &_module);
    uai_module_launch(_module);
    _batch_size = device_cfg->getUaiBatchSize();
    for (int i = 0; i < device_cfg->getUaiQueueDepth(); i++) {
      _payloads.emplace_back(_module, _batch_size);
    }
  }

  virtual int Inference(std::vector<Sample> samples) {
    if (_payloads_queue.Full(MSQ::Stage::FREE)) {
      return -1;
    }
    TRACE_BATCH_EVENT_INSTANT_ON_DEVICE(at_device, kilt_utils::GetSampleID(samples[0]), _device_id);
    _n_scheduled_inferences += _batch_size;
    auto& p = _payloads_queue.GetFirst(MSQ::Stage::FREE);
    p.samples = samples;
    _payloads_queue.Push(MSQ::Stage::HAS_SAMPLES);

    if (_flush_immediately) {
      // Wait for a free payload
      while (_payloads_queue.Full(MSQ::Stage::FREE)) {
        if (_flush_yield_time)
          std::this_thread::sleep_for(std::chrono::microseconds(_flush_yield_time));
      }

      auto& p = _payloads_queue.GetFirst(MSQ::Stage::FREE);
      p.ignore = true;
      _payloads_queue.Push(MSQ::Stage::HAS_SAMPLES);
    }

    return 1;
  }

  virtual void Flush() {
    // Only do this once
    {
      std::scoped_lock l(_flushed_lock);
      if (_flushed) {
        return;
      }
      _flushed = true;
    }

    std::this_thread::sleep_for(std::chrono::microseconds(_before_flush_wait_time));

    size_t n_left = _device_reset_sample_count - (_n_scheduled_inferences % _device_reset_sample_count);
    if (n_left == _device_reset_sample_count) {
      return;
    }

    for (auto i = 0; i < n_left / _batch_size; i++) {
      TRACE_EVENT_ON_DEVICE(flush_wait_for_payload, _device_id);
      while (_payloads_queue.Full(MSQ::Stage::FREE)) {
        if (_scheduler_yield_time)
          std::this_thread::sleep_for(std::chrono::microseconds(_scheduler_yield_time));
      }
      TRACE_EVENT_END(flush_wait_for_payload);
      TRACE_EVENT_INSTANT_ON_DEVICE(flush_device, _device_id);

      auto& p = _payloads_queue.GetFirst(MSQ::Stage::FREE);
      p.ignore = true;
      _payloads_queue.Push(MSQ::Stage::HAS_SAMPLES);
    }
  }

#ifdef TESTING
  bool done() {
    return _completed == 2;
  }
#endif
 private:
  using MSQ = MultiStageQueue<Payload<Sample>>;

  void QueueScheduler() {
    LONG_TRACE_EVENT_ON_DEVICE_BEGIN(scheduler_wait_for_batch, _device_id);
    while (!_terminate) {
      if (_payloads_queue.Full(MSQ::Stage::HAS_SAMPLES)) {
        if (_scheduler_yield_time)
          std::this_thread::sleep_for(std::chrono::microseconds(_scheduler_yield_time));
        continue;
      }
      LONG_TRACE_EVENT_ON_DEVICE_END(scheduler_wait_for_batch, _device_id);

      auto& payload = _payloads_queue.GetFirst(MSQ::Stage::HAS_SAMPLES);
      OneShot(payload);  // Might want to wait for the device to be free, otherwise we *could* get contention

      _payloads_queue.Push(MSQ::Stage::ON_DEVICE);

      LONG_TRACE_EVENT_ON_DEVICE_BEGIN(scheduler_wait_for_batch, _device_id);
      LONG_TRACE_BATCH_EVENT_ON_DEVICE_BEGIN(device_inference, kilt_utils::GetSampleID(payload.samples[0]), _device_id);
    }
    LONG_TRACE_EVENT_ON_DEVICE_END(scheduler_wait_for_batch, _device_id);
  }

  void OneShot(Payload<Sample>& p) {
    TRACE_BATCH_EVENT_ON_DEVICE(one_shot, kilt_utils::GetSampleID(p.samples[0]), _device_id);

    // ConfigureWorkload
    const std::vector<Sample> samples = p.samples;
    std::vector<void*> buffers = p.GetBuffers().GetRawInputBuffers();

    TRACE_BATCH_EVENT_ON_DEVICE(configure_workload, kilt_utils::GetSampleID(p.samples[0]), _device_id);
    _model->configureWorkload(_data_source, this, &samples, buffers);
    TRACE_EVENT_END(configure_workload);

    TRACE_BATCH_EVENT_ON_DEVICE(enqueue, kilt_utils::GetSampleID(p.samples[0]), _device_id);
    p.Enqueue(_module);
    TRACE_EVENT_END(enqueue);
  }

#if SIMD_ARGMAX
#pragma message("Using SIMD Argmax")
  void fast_fp8_fused_dequantise_argmax(const uint8_t* _buffer, std::vector<int64_t>& bufs) {
    // Constants used to "dequantise" the quantized values
    const __m256i boundary = _mm256_setzero_si256();
    const __m256i xor_mask = _mm256_set1_epi8(0b01111111);

    for (size_t i = 0; i < bufs.size(); i++) {
      const __m256i plus32 = _mm256_set1_epi8(1);  // Remember to convert this back to 32

      // Track the current values' indices
      // But an 8 bit integer can only represent indices upto 127, whilst our indices can reach 1023
      // To fix this, we'll use 2 indices, the 32 byte index, and an offset into the 32 byte vector
      // idx stores the 32 byte index, and the position of an index within idx, represents the offset
      // So to get the actual index, we need idx[k] * 32 + k
      __m256i idx = _mm256_setzero_si256();

      // Track the maximum values and their indices - indices are as above
      __m256i max_val_v = _mm256_set1_epi8(-128);
      __m256i max_idx_v = _mm256_setzero_si256();

      for (size_t j = 0; j < 1024; j += 32) {
        // Load 32 values from the input buffer into a 256-bit vector.
        const __m256i* src_ptr = reinterpret_cast<const __m256i*>(_buffer + 1024 * i + j);
        __m256i qres_v = _mm256_stream_load_si256(src_ptr);

        // XOR all values that are less than 0 with 0b0111_1111.
        // Now if v1 < v2 then deQuant(v1) < deQuant(v2).
        __m256i mask = _mm256_cmpgt_epi8(boundary, qres_v);
        mask = _mm256_and_si256(mask, xor_mask);
        qres_v = _mm256_xor_si256(qres_v, mask);

        // Update the maximum values and their indices.
        const __m256i gt = _mm256_cmpgt_epi8(qres_v, max_val_v);
        max_idx_v = _mm256_blendv_epi8(max_idx_v, idx, gt);
        max_val_v = _mm256_max_epi8(max_val_v, qres_v);

        // Update all current index
        idx = _mm256_add_epi8(idx, plus32);
      }

      // Get the vector values out
      int8_t max_arr[32], idx_arr[32];
      _mm256_storeu_si256((__m256i*)max_arr, max_val_v);
      _mm256_storeu_si256((__m256i*)idx_arr, max_idx_v);

      // Argmax over the vector
      int8_t max_val = 0;
      size_t max_idx = 0;

      for (int64_t k = 0; k < 32; k++) {
        const size_t idx = idx_arr[k] * 32 + k;
        const int8_t val = max_arr[k];

        if (val > max_val) {
          max_idx = idx;
          max_val = val;
        }
      }
      bufs[i] = max_idx;
    }
  }
#else
  void fast_fp8_fused_dequantise_argmax(const uint8_t* _buffer, std::vector<int64_t>& bufs) {
    for (size_t i = 0; i < bufs.size(); i++) {
      int64_t max_idx = 0;
      int8_t max_val = -128;

      for (int64_t j = 0; j < 1024; j++) {
        int8_t qres = (int8_t)_buffer[1024 * i + j];

        // XOR all values that are less than 0 with 0b0111_1111.
        // Now if v1 < v2 then deQuant(v1) < deQuant(v2).
        if (qres < 0) {
          qres = qres ^ 0b01111111;
        }

        if (qres > max_val) {
          max_idx = j;
          max_val = qres;
        }
      }
      bufs[i] = max_idx;
    }
  }
#endif
  void fill_argmax(const Payload<Sample>& payload, std::vector<int64_t>& bufs) {
    uint8_t* buf = (uint8_t*)payload.GetBuffers().GetRawOutputBuffers()[0];

#if SIMD_ARGMAX
    fast_fp8_fused_dequantise_argmax(buf, bufs);
#else
    this->SyncData(buf, _buffer.data(), 0, 1024 * bufs.size());
    fast_fp8_fused_dequantise_argmax(_buffer.data(), bufs);
#endif
  }

  void PostProcessor() {
    while (!_terminate) {
      if (_payloads_queue.Full(MSQ::Stage::ON_DEVICE)) {
        if (_postprocessor_yield_time)
          std::this_thread::sleep_for(std::chrono::microseconds(_postprocessor_yield_time));
        continue;
      }

      auto& payload = _payloads_queue.GetFirst(MSQ::Stage::ON_DEVICE);

      payload.Wait(_module, _wait_timeout);
      LONG_TRACE_BATCH_EVENT_ON_DEVICE_END(device_inference, kilt_utils::GetSampleID(payload.samples[0]), _device_id);

      TRACE_BATCH_EVENT(postprocess, kilt_utils::GetSampleID(payload.samples[0]));
      auto bufs = payload.GetBuffers().GetRawOutputBuffers();

      std::vector<int64_t> argmaxes(_batch_size);
      std::vector<void*> out_bufs(_batch_size);

      for (size_t i = 0; i < _batch_size; i++) {
        out_bufs[i] = static_cast<void*>(&argmaxes[i]);
      }

      TRACE_BATCH_EVENT(argmax, kilt_utils::GetSampleID(payload.samples[0]));
      fill_argmax(payload, argmaxes);
      TRACE_EVENT_END(argmax);

      if (!payload.ignore) [[likely]] {
        _model->postprocessResults(&(payload.samples), out_bufs);
      }

      TRACE_EVENT_END(postprocess);

      _payloads_queue.Push(MSQ::Stage::FREE);

#ifdef TESTING
      _completed++;
#endif
    }
  }

  UaiModule* _module = NULL;
  UaiEvent _event;

  std::mutex _flushed_lock;
  bool _flushed = false;

  std::atomic<bool> _terminate = false;
#ifdef TESTING
  std::atomic<int> _completed = 0;
#endif

  int _device_id = -1;

  int _scheduler_yield_time = 10;
  int _postprocessor_yield_time = 10;
  int _wait_timeout = -1;
  size_t _batch_size;

  size_t _n_scheduled_inferences = 0;
  size_t _device_reset_sample_count = 100;

  std::vector<Payload<Sample>> _payloads;
  MultiStageQueue<Payload<Sample>> _payloads_queue;

  int _before_flush_wait_time = 5000000;
  int _flush_yield_time = 10;
  bool _flush_immediately = false;

#if !(SIMD_ARGMAX)
  std::vector<uint8_t> _buffer;
#endif

  IModel* _model;
  IDataSource* _data_source;

  std::vector<std::thread> _threads;

  State _state;
};

template <typename Sample>
IDevice<Sample>* createDevice(IModel* _model, IDataSource* _data_source, IConfig* _config, int hw_id, std::vector<int> aff) {
  UAIDeviceConfig* device_cfg = static_cast<UAIDeviceConfig*>(_config->device_cfg);
  Device<Sample>* d = new Device<Sample>();
  d->Construct(_model, _data_source, device_cfg, hw_id, aff);
  return d;
}
