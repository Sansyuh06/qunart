# Deploying a qunart-compressed Model on Android

This guide covers running a GGUF-exported model on an Android phone using llama.cpp.

## Prerequisites

- A qunart-compressed model exported to GGUF format (see `--export gguf`)
- Android phone with at least 4 GB RAM (8+ GB recommended)
- Android Studio (for building the JNI app) or a pre-built llama.cpp Android binary

## Option 1: llama.cpp Android App (Recommended)

### Build llama.cpp for Android

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Set up Android NDK (download from https://developer.android.com/ndk)
export ANDROID_NDK=/path/to/android-ndk

# Build for ARM64
mkdir build-android && cd build-android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j $(nproc)
```

### Transfer and Run

```bash
# Push binary and model to phone
adb push build-android/bin/llama-cli /data/local/tmp/
adb push model-Q4_K_M.gguf /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x llama-cli
./llama-cli -m model-Q4_K_M.gguf \
  -p "What is machine learning?" \
  -n 128 \
  --threads 4
```

### Measure Performance

```bash
# Tokens per second and peak RAM are printed by llama-cli:
#   llama_print_timings: eval time = ... ms / ... tokens (... ms per token, ... tokens per second)
# Peak RAM: monitor via `adb shell dumpsys meminfo <pid>`

# For systematic benchmarking:
./llama-cli -m model-Q4_K_M.gguf \
  -p "Explain quantum computing in simple terms." \
  -n 256 \
  --threads 4 2>&1 | grep -E "(eval time|load time|sample time|total time)"
```

## Option 2: llama.rn (React Native)

For a polished mobile app:

```bash
npx react-native init QunartApp
cd QunartApp
npm install llama.rn
```

See [llama.rn documentation](https://github.com/mybigday/llama.rn) for integration details.

Key API:
```javascript
import { initLlama } from 'llama.rn';

const context = await initLlama({
  model: 'file:///path/to/model-Q4_K_M.gguf',
  n_ctx: 2048,
  n_threads: 4,
});

const result = await context.completion({
  prompt: 'Hello, how are you?',
  n_predict: 128,
});
console.log(result.text);
```

## Option 3: MLC LLM (Alternative)

For WebGPU/Vulkan acceleration on supported devices:

```bash
pip install mlc-llm
mlc_llm convert_weight model-dir/ -o dist/
mlc_llm gen_config model-dir/ -o dist/ --quantization q4f16_1
```

## Verifying Offline Operation

1. Transfer the GGUF file to the phone
2. Enable airplane mode
3. Disable WiFi and Bluetooth
4. Run the model — it should work entirely offline
5. Verify no network calls in `adb logcat | grep -i network`

## Performance Expectations

| Model Size (Q4) | RAM Required | Expected tok/s (SD 888+) |
|-----------------|-------------|-------------------------|
| < 500 MB        | ~1.5 GB     | 15-25 tok/s            |
| 500 MB - 1 GB   | ~2.5 GB     | 8-15 tok/s             |
| 1 - 2 GB        | ~4 GB       | 4-8 tok/s              |

*Actual numbers depend on the device. Measure on your target hardware.*

## Constraints for Pruned Models

- `hidden_size` and `intermediate_size` should be multiples of 64 for GGUF compatibility
- `num_attention_heads` should ideally be a power of 2
- If GGUF conversion fails, qunart's planner can pad dimensions to meet these constraints
  (use `--pad-for-export` flag, documented in the CLI help)

## Troubleshooting

- **"unsupported model architecture"**: Ensure the pruned model's config.json has the correct `architectures` field
- **OOM on phone**: Try a more aggressive quantization (Q3_K_S) or reduce context length
- **Slow generation**: Reduce `--threads` to match physical cores (not hyperthreaded)
