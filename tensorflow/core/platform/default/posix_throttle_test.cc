/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/default/posix_throttle.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

class TestTime : public EnvTime {
 public:
  uint64 GetOverridableNowNanos() const override {
    return now_micros_ * kMicrosToNanos;
  }

  void SetTime(uint64 now_micros) { now_micros_ = now_micros; }

  void AdvanceSeconds(int64 secs) { now_micros_ += secs * kSecondsToMicros; }

 private:
  uint64 now_micros_ = 1234567890000000ULL;
};

class PosixThrottleTest : public ::testing::Test {
 protected:
  PosixThrottleTest() : throttle_(&time_) {
    config_.enabled = true;
    config_.tokens_per_request = 100;  // NOTE(mkuchnik): assumed limit
    throttle_.SetConfig(config_);
  }

  PosixThrottleConfig config_;
  TestTime time_;
  PosixThrottle throttle_;
};

TEST_F(PosixThrottleTest, ReplenishTokens) {
  EXPECT_EQ(0, throttle_.available_tokens());
  time_.AdvanceSeconds(1);
  EXPECT_EQ(100000, throttle_.available_tokens());
  time_.AdvanceSeconds(2);
  EXPECT_EQ(300000, throttle_.available_tokens());
}

TEST_F(PosixThrottleTest, RejectRequest) {
  EXPECT_EQ(0, throttle_.available_tokens());
  time_.AdvanceSeconds(1);
  EXPECT_TRUE(throttle_.AdmitRequest());
  EXPECT_EQ(99900, throttle_.available_tokens());
  for (int i = 1; i < 1000; i++) {
    EXPECT_TRUE(throttle_.AdmitRequest());
  }
  EXPECT_FALSE(throttle_.AdmitRequest());
}

TEST_F(PosixThrottleTest, MarkResponses) {
  time_.AdvanceSeconds(1);
  EXPECT_TRUE(throttle_.AdmitRequest());
  throttle_.RecordResponse(128000000);  // 128 MB response
  EXPECT_EQ(-25100, throttle_.available_tokens());
  EXPECT_FALSE(throttle_.AdmitRequest());
  time_.AdvanceSeconds(1);
  EXPECT_TRUE(throttle_.AdmitRequest())
      << "Available tokens: " << throttle_.available_tokens();
}

TEST_F(PosixThrottleTest, Skippingtime_) {
  EXPECT_EQ(0, throttle_.available_tokens());
  time_.AdvanceSeconds(90);
  EXPECT_EQ(9000000, throttle_.available_tokens());
}

TEST_F(PosixThrottleTest, BucketLimit) {
  time_.AdvanceSeconds(120);
  EXPECT_EQ(10000000, throttle_.available_tokens());
}

TEST_F(PosixThrottleTest, ReverseTime) {
  time_.AdvanceSeconds(1);
  EXPECT_EQ(100000, throttle_.available_tokens());
  time_.AdvanceSeconds(-3600);
  EXPECT_EQ(100000, throttle_.available_tokens());
  time_.AdvanceSeconds(1);
  EXPECT_EQ(200000, throttle_.available_tokens());
}

TEST(PosixThrottleDisabledTest, Disabled) {
  TestTime time;
  PosixThrottle throttle(&time);
  ASSERT_FALSE(throttle.is_enabled());  // Verify throttle is disabled.

  EXPECT_EQ(0, throttle.available_tokens());
  time.AdvanceSeconds(1);
  EXPECT_EQ(100000, throttle.available_tokens());
  EXPECT_TRUE(throttle.AdmitRequest());
  EXPECT_EQ(99900, throttle.available_tokens());
  time.AdvanceSeconds(1);
  EXPECT_EQ(199900, throttle.available_tokens());
  throttle.RecordResponse(128000000);  // 128 MB response.
  EXPECT_LT(0, throttle.available_tokens());
  // Admit request even without available tokens
  EXPECT_TRUE(throttle.AdmitRequest());
}

}  // namespace

}  // namespace tensorflow
