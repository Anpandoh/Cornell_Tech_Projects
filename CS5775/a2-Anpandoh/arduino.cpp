/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <TensorFlowLite.h>

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "main_functions.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Define PROFILE_MICRO_SPEECH to enable detailed profiling
#define PROFILE_MICRO_SPEECH

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 10 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
static tflite::MicroMutableOpResolver<6> micro_op_resolver; 
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddTranspose() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddPad() != kTfLiteOk) {
    return;
  }


  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    MicroPrintf("Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  previous_time = 0;

  // start the audio
  TfLiteStatus init_status = InitAudioRecording();
  if (init_status != kTfLiteOk) {
    MicroPrintf("Unable to initialize audio");
    return;
  }

  MicroPrintf("Initialization complete");
}

// The name of this function is important for Arduino compatibility.
void loop() {
#ifdef PROFILE_MICRO_SPEECH
  const uint32_t prof_start_total = millis();
  static uint32_t prof_count = 0;
  
  // Total time
  static uint32_t prof_sum_total = 0;
  static uint32_t prof_min_total = std::numeric_limits<uint32_t>::max();
  static uint32_t prof_max_total = 0;
  
  // Preprocessing time
  static uint32_t prof_sum_preproc = 0;
  static uint32_t prof_min_preproc = std::numeric_limits<uint32_t>::max();
  static uint32_t prof_max_preproc = 0;
  
  // Neural network time
  static uint32_t prof_sum_nn = 0;
  static uint32_t prof_min_nn = std::numeric_limits<uint32_t>::max();
  static uint32_t prof_max_nn = 0;
  
  // Postprocessing time
  static uint32_t prof_sum_postproc = 0;
  static uint32_t prof_min_postproc = std::numeric_limits<uint32_t>::max();
  static uint32_t prof_max_postproc = 0;
  
  uint32_t stage_start, stage_end, elapsed;
#endif  // PROFILE_MICRO_SPEECH

  // PREPROCESSING STAGE
#ifdef PROFILE_MICRO_SPEECH
  stage_start = millis();
#endif

  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    MicroPrintf("Feature generation failed");
    return;
  }
  previous_time += how_many_new_slices * kFeatureSliceStrideMs;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

#ifdef PROFILE_MICRO_SPEECH
  stage_end = millis();
  elapsed = stage_end - stage_start;
  prof_sum_preproc += elapsed;
  if (elapsed < prof_min_preproc) prof_min_preproc = elapsed;
  if (elapsed > prof_max_preproc) prof_max_preproc = elapsed;
#endif

  // NEURAL NETWORK STAGE
#ifdef PROFILE_MICRO_SPEECH
  stage_start = millis();
#endif

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

#ifdef PROFILE_MICRO_SPEECH
  stage_end = millis();
  elapsed = stage_end - stage_start;
  prof_sum_nn += elapsed;
  if (elapsed < prof_min_nn) prof_min_nn = elapsed;
  if (elapsed > prof_max_nn) prof_max_nn = elapsed;
#endif

  // POSTPROCESSING STAGE
#ifdef PROFILE_MICRO_SPEECH
  stage_start = millis();
#endif

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    MicroPrintf("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  RespondToCommand(current_time, found_command, score, is_new_command);

#ifdef PROFILE_MICRO_SPEECH
  stage_end = millis();
  elapsed = stage_end - stage_start;
  prof_sum_postproc += elapsed;
  if (elapsed < prof_min_postproc) prof_min_postproc = elapsed;
  if (elapsed > prof_max_postproc) prof_max_postproc = elapsed;

  const uint32_t prof_end_total = millis();
  if (++prof_count > 10) {
    uint32_t elapsed_total = prof_end_total - prof_start_total;
    prof_sum_total += elapsed_total;
    if (elapsed_total < prof_min_total) {
      prof_min_total = elapsed_total;
    }
    if (elapsed_total > prof_max_total) {
      prof_max_total = elapsed_total;
    }
    if (prof_count % 300 == 0) {
      MicroPrintf("## Total time: min %dms  max %dms  avg %dms", 
                  prof_min_total, prof_max_total, prof_sum_total / prof_count);
      MicroPrintf("## Preprocessing: min %dms  max %dms  avg %dms", 
                  prof_min_preproc, prof_max_preproc, prof_sum_preproc / prof_count);
      MicroPrintf("## Neural Network: min %dms  max %dms  avg %dms", 
                  prof_min_nn, prof_max_nn, prof_sum_nn / prof_count);
      MicroPrintf("## Postprocessing: min %dms  max %dms  avg %dms", 
                  prof_min_postproc, prof_max_postproc, prof_sum_postproc / prof_count);
    }
  }
#endif  // PROFILE_MICRO_SPEECH
}
