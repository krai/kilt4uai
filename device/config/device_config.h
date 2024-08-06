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

#include "config/config_tools/config_tools.h"
#include "iconfig.h"

namespace KRAI {

class UAIDeviceConfig : public IDeviceConfig {
 public:
  virtual const std::string getModelRoot() const { return uai_model_root; }

  const size_t getUaiQueueDepth() const {
    return uai_queue_depth;
  }

  const int getUaiSchedulerYieldTime() const {
    return uai_scheduler_yield_time;
  }

  const int getUaiFlushYieldTime() const {
    return uai_flush_yield_time;
  }

  const int getUaiFlushImmediately() const {
    return uai_flush_immediately;
  }

  const int getUaiPostprocessorYieldTime() const {
    return uai_postprocessor_yield_time;
  }

  const int getUaiModelBatchSize() const {
    return uai_model_batch_size;
  }

  const int getUaiBatchSize() const {
    return uai_batch_size;
  }

  const int getBeforeFlushWaitTime() const {
    return before_flush_wait_time;
  }

  const int getUaiWaitTimeout() const {
    return uai_wait_timeout;
  }

 private:
  const char* uai_model_root = getconfig_c("KILT_MODEL_ROOT");
  const size_t uai_queue_depth = alter_str_i(getconfig_c("KILT_DEVICE_UAI_QUEUE_LENGTH"), 1);
  const int uai_scheduler_yield_time = alter_str_i(getconfig_c("KILT_DEVICE_UAI_SCHEDULER_YIELD_TIME"), 0);
  const int uai_flush_yield_time = alter_str_i(getconfig_c("KILT_DEVICE_UAI_FLUSH_YIELD_TIME"), 0);
  const bool uai_flush_immediately = getconfig_b("KILT_DEVICE_UAI_FLUSH_IMMEDIATELY");
  const int uai_postprocessor_yield_time = alter_str_i(getconfig_c("KILT_DEVICE_UAI_POSTPROCESSOR_TIME"), 10);
  const int uai_model_batch_size = alter_str_i(getconfig_c("KILT_DEVICE_UAI_MODEL_BATCH_SIZE"), 100);
  const size_t uai_batch_size = getconfig_i("KILT_MODEL_BATCH_SIZE");
  const int before_flush_wait_time = alter_str_i(getconfig_c("KILT_DEVICE_UAI_MODEL_BEFORE_FLUSH_WAIT_TIME"), 5000000);
  const int uai_wait_timeout = alter_str_i(getconfig_c("KILT_DEVICE_UAI_WAIT_TIMEOUT"), -1);
};

IDeviceConfig* getDeviceConfig() { return new UAIDeviceConfig(); }

};  // namespace KRAI
