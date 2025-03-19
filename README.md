[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Microsoft/DeepScale/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/deepscale.svg)](https://pypi.org/project/deepscale/)
[![Downloads](https://static.pepy.tech/badge/deepscale)](https://pepy.tech/project/deepscale)
[![Build](https://badgen.net/badge/build/check-status/blue)](#build-pipeline-status)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9530/badge)](https://www.bestpractices.dev/projects/9530)
[![Twitter](https://img.shields.io/twitter/follow/MSFTDeepScale)](https://twitter.com/intent/follow?screen_name=MSFTDeepScale)
[![Japanese Twitter](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9ETwitter-%40MSFTDeepScaleJP-blue)](https://twitter.com/MSFTDeepScaleJP)
[![Chinese Zhihu](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%BE%AE%E8%BD%AFDeepScale-blue)](https://www.zhihu.com/people/deepscale)

<!-- NOTE: we must use html for news items otherwise links will be broken in the 'more news' section -->
<details>
 <summary>Nnews</summary>
 <ul>
  <li>[2023/08] <a href="https://github.com/khulnasoft/DeepScaleExamples/blob/master/inference/huggingface/zero_inference/README.md">DeepScale ZeRO-Inference: 20x faster inference through weight quantization and KV cache offloading</a></li>

  <li>[2023/08] <a href="https://github.com/khulnasoft/DeepScale/tree/master/blogs/deepscale-chat/ds-chat-release-8-31/README.md">DeepScale-Chat: Llama/Llama-2 system support, efficiency boost, and training stability improvements</a></li>

  <li>[2023/08] <a href="https://github.com/khulnasoft/DeepScale/tree/master/blogs/deepscale-ulysses">DeepScale Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models</a> [<a href="https://github.com/khulnasoft/DeepScale/blob/master/blogs/deepscale-ulysses/chinese/README.md">中文</a>] [<a href="https://github.com/khulnasoft/DeepScale/blob/master/blogs/deepscale-ulysses/japanese/README.md">日本語</a>]</li>

  <li>[2023/06] <a href="https://www.microsoft.com/en-us/research/blog/deepscale-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/">ZeRO++: A leap in speed for LLM and chat model training with 4X less communication</a> [<a href="https://www.microsoft.com/en-us/research/blog/deepscale-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/">English</a>] [<a href="https://github.com/khulnasoft/DeepScale/blob/master/blogs/zeropp/chinese/README.md">中文</a>] [<a href="https://github.com/khulnasoft/DeepScale/blob/master/blogs/zeropp/japanese/README.md">日本語</a>]</li>
 </ul>
</details>

---

# Extreme Speed and Scale for DL Training and Inference

***[DeepScale](https://www.deepscale.khulnasoft.com/) enables world's most powerful language models like [MT-530B](https://www.microsoft.com/en-us/research/blog/using-deepscale-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) and [BLOOM](https://huggingface.co/blog/bloom-megatron-deepscale)***. It is an easy-to-use deep learning optimization software suite that powers unprecedented scale and speed for both training and inference. With DeepScale you can:

* Train/Inference dense or sparse models with billions or trillions of parameters
* Achieve excellent system throughput and efficiently scale to thousands of GPUs
* Train/Inference on resource constrained GPU systems
* Achieve unprecedented low latency and high throughput for inference
* Achieve extreme compression for an unparalleled inference latency and model size reduction with low costs

---

# DeepScale's four innovation pillars

<img src="docs/assets/images/DeepScale-pillars.png" width="800px">


## DeepScale-Compression

To further increase the inference efficiency, DeepScale offers easy-to-use and flexible-to-compose compression techniques for researchers and practitioners to compress their models while delivering faster speed, smaller model size, and significantly reduced compression cost. Moreover, SoTA innovations on compression like ZeroQuant and XTC are included under the compression pillar. Learn more: [DeepScale-Compression](https://www.deepscale.khulnasoft.com/compression)

## DeepScale4Science

In line with Microsoft's mission to solve humanity's most pressing challenges, the DeepScale team at Microsoft is responding to this opportunity by launching a new initiative called *DeepScale4Science*, aiming to build unique capabilities through AI system technology innovations to help domain experts to unlock today's biggest science mysteries. Learn more: [DeepScale4Science website](https://deepscale4science.ai/) and [tutorials](https://www.deepscale.khulnasoft.com/deepscale4science/)

---

# DeepScale Software Suite

## DeepScale Library

   The [DeepScale](https://github.com/khulnasoft/deepscale) library (this repository) implements and packages the innovations and technologies in DeepScale Training, Inference and Compression Pillars into a single easy-to-use, open-sourced repository. It allows for easy composition of multitude of features within a single training, inference or compression pipeline. The DeepScale Library is heavily adopted by the DL community, and has been used to enable some of the most powerful models (see [DeepScale Adoption](#deepscale-adoption)).

## Model Implementations for Inference (MII)

   [Model Implementations for Inference (MII)](https://github.com/khulnasoft/deepscale-mii) is an open-sourced repository for making low-latency and high-throughput inference accessible to all data scientists by alleviating the need to apply complex system optimization techniques themselves. Out-of-box, MII offers support for thousands of widely used DL models, optimized using DeepScale-Inference, that can be deployed with a few lines of code, while achieving significant latency reduction compared to their vanilla open-sourced versions.

## DeepScale on Azure

   DeepScale users are diverse and have access to different environments. We recommend to try DeepScale on Azure as it is the simplest and easiest method. The recommended method to try DeepScale on Azure is through AzureML [recipes](https://github.com/Azure/azureml-examples/tree/main/v1/python-sdk/workflows/train/deepscale). The job submission and data preparation scripts have been made available [here](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepscale/azureml). For more details on how to use DeepScale on Azure, please follow the [Azure tutorial](https://www.deepscale.khulnasoft.com/tutorials/azure/).

---

DeepScale has been used to train many different large-scale models, below is a list of several examples that we are aware of (if you'd like to include your model please submit a PR):

  * [Megatron-Turing NLG (530B)](https://www.microsoft.com/en-us/research/blog/using-deepscale-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)
  * [Jurassic-1 (178B)](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)
  * [BLOOM (176B)](https://huggingface.co/blog/bloom-megatron-deepscale)
  * [GLM (130B)](https://github.com/THUDM/GLM-130B)
  * [xTrimoPGLM (100B)](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v2)
  * [YaLM (100B)](https://github.com/yandex/YaLM-100B)
  * [GPT-NeoX (20B)](https://github.com/EleutherAI/gpt-neox)
  * [AlexaTM (20B)](https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning)
  * [Turing NLG (17B)](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)
  * [METRO-LM (5.4B)](https://arxiv.org/pdf/2204.06644.pdf)

DeepScale has been integrated with several different popular open-source DL frameworks such as:

|                                                                                                | Documentation                                |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------- |
<img src="docs/assets/images/transformers-light.png#gh-light-mode-only" width="250px"><img src="docs/assets/images/transformers-dark.png#gh-dark-mode-only" width="250px"> | [Transformers with DeepScale](https://huggingface.co/docs/transformers/main/main_classes/deepscale) |
| <img src="docs/assets/images/accelerate-light.png#gh-light-mode-only" width="250px"><img src="docs/assets/images/accelerate-dark.png#gh-dark-mode-only" width="250px"> | [Accelerate with DeepScale](https://huggingface.co/docs/accelerate/usage_guides/deepscale) |
| <img src="docs/assets/images/lightning-light.svg#gh-light-mode-only" width="200px"><img src="docs/assets/images/lightning-dark.svg#gh-dark-mode-only" width="200px"> | [Lightning with DeepScale](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#deepscale) |
| <img src="docs/assets/images/mosaicml.svg" width="200px"> | [MosaicML with DeepScale](https://docs.mosaicml.com/projects/composer/en/latest/trainer/using_the_trainer.html?highlight=deepscale#deepscale-integration) |
| <img src="docs/assets/images/determined.svg" width="225px"> | [Determined with DeepScale](https://docs.determined.ai/latest/training/apis-howto/deepscale/overview.html) |
| <img src="https://user-images.githubusercontent.com/58739961/187154444-fce76639-ac8d-429b-9354-c6fac64b7ef8.jpg" width=150> | [MMEngine with DeepScale](https://mmengine.readthedocs.io/en/latest/common_usage/large_model_training.html#deepscale) |

---

# Build Pipeline Status

| Description | Status |
| ----------- | ------ |
| NVIDIA | [![nv-torch110-p40](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-torch110-p40.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-torch110-p40.yml) [![nv-torch110-v100](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-torch110-v100.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-torch110-v100.yml) [![nv-torch-latest-v100](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-torch-latest-v100.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-torch-latest-v100.yml) [![nv-h100](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-h100.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-h100.yml) [![nv-inference](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-inference.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-inference.yml) [![nv-nightly](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-nightly.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-nightly.yml) |
| AMD | [![amd-mi200](https://github.com/khulnasoft/DeepScale/actions/workflows/amd-mi200.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/amd-mi200.yml) |
| CPU | [![torch-latest-cpu](https://github.com/khulnasoft/DeepScale/actions/workflows/cpu-torch-latest.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/cpu-torch-latest.yml) [![cpu-inference](https://github.com/khulnasoft/DeepScale/actions/workflows/cpu-inference.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/cpu-inference.yml) |
| Intel Gaudi | [![hpu-gaudi2](https://github.com/khulnasoft/DeepScale/actions/workflows/hpu-gaudi2.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/hpu-gaudi2.yml) |
| Intel XPU | [![xpu-max1100](https://github.com/khulnasoft/DeepScale/actions/workflows/xpu-max1100.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/xpu-max1100.yml) |
| PyTorch Nightly | [![nv-torch-nightly-v100](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-torch-nightly-v100.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-torch-nightly-v100.yml) |
| Integrations | [![nv-transformers-v100](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-transformers-v100.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-transformers-v100.yml) [![nv-lightning-v100](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-lightning-v100.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-lightning-v100.yml) [![nv-accelerate-v100](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-accelerate-v100.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-accelerate-v100.yml) [![nv-mii](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-mii.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-mii.yml) [![nv-ds-chat](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-ds-chat.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-ds-chat.yml) [![nv-sd](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-sd.yml/badge.svg)](https://github.com/khulnasoft/DeepScale/actions/workflows/nv-sd.yml) |
| Misc | [![Formatting](https://github.com/khulnasoft/DeepScale/actions/workflows/formatting.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/formatting.yml) [![pages-build-deployment](https://github.com/khulnasoft/DeepScale/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/khulnasoft/DeepScale/actions/workflows/pages/pages-build-deployment) [![Documentation Status](https://readthedocs.org/projects/deepscale/badge/?version=latest)](https://deepscale.readthedocs.io/en/latest/?badge=latest)[![python](https://github.com/khulnasoft/DeepScale/actions/workflows/python.yml/badge.svg?branch=master)](https://github.com/khulnasoft/DeepScale/actions/workflows/python.yml) |
| Huawei Ascend NPU | [![Huawei Ascend NPU](https://github.com/cosdt/DeepScale/actions/workflows/huawei-ascend-npu.yml/badge.svg?branch=master)](https://github.com/cosdt/DeepScale/actions/workflows/huawei-ascend-npu.yml) |

# Installation

The quickest way to get started with DeepScale is via pip, this will install
the latest release of DeepScale which is not tied to specific PyTorch or CUDA
versions. DeepScale includes several C++/CUDA extensions that we commonly refer
to as our 'ops'.  By default, all of these extensions/ops will be built
just-in-time (JIT) using [torch's JIT C++ extension loader that relies on
ninja](https://pytorch.org/docs/stable/cpp_extension.html) to build and
dynamically link them at runtime.

## Requirements
* [PyTorch](https://pytorch.org/) must be installed _before_ installing DeepScale.
* For full feature support we recommend a version of PyTorch that is >= 1.9 and ideally the latest PyTorch stable release.
* A CUDA or ROCm compiler such as [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#introduction) or [hipcc](https://github.com/ROCm-Developer-Tools/HIPCC) used to compile C++/CUDA/HIP extensions.
* Specific GPUs we develop and test against are listed below, this doesn't mean your GPU will not work if it doesn't fall into this category it's just DeepScale is most well tested on the following:
  * NVIDIA: Pascal, Volta, Ampere, and Hopper architectures
  * AMD: MI100 and MI200

## Contributed HW support
* DeepScale now support various HW accelerators.

| Contributor | Hardware                            | Accelerator Name | Contributor validated | Upstream validated |
|-------------|-------------------------------------|------------------| --------------------- |--------------------|
| Huawei      | Huawei Ascend NPU                   | npu              | Yes | No                 |
| Intel       | Intel(R) Gaudi(R) 2 AI accelerator  | hpu              | Yes | Yes                |
| Intel       | Intel(R) Xeon(R) Processors         | cpu              | Yes | Yes                |
| Intel       | Intel(R) Data Center GPU Max series | xpu              | Yes | Yes                |

## PyPI
We regularly push releases to [PyPI](https://pypi.org/project/deepscale/) and encourage users to install from there in most cases.

```bash
pip install deepscale
```

After installation, you can validate your install and see which extensions/ops
your machine is compatible with via the DeepScale environment report.

```bash
ds_report
```

If you would like to pre-install any of the DeepScale extensions/ops (instead
of JIT compiling) or install pre-compiled ops via PyPI please see our [advanced
installation instructions](https://www.deepscale.khulnasoft.com/tutorials/advanced-install/).

## Windows
Windows support is partially supported with DeepScale. On Windows you can build wheel with following steps, currently only inference mode is supported.
1. Install pytorch, such as pytorch 1.8 + cuda 11.1
2. Install visual cpp build tools, such as VS2019 C++ x64/x86 build tools
3. Launch cmd console with Administrator privilege for creating required symlink folders
4. Run `python setup.py bdist_wheel` to build wheel in `dist` folder

# Features

Please checkout [DeepScale-Training](https://www.deepscale.khulnasoft.com/training), [DeepScale-Inference](https://www.deepscale.khulnasoft.com/inference) and [DeepScale-Compression](https://www.deepscale.khulnasoft.com/compression) pages for full set of features offered along each of these three pillars.

# Further Reading

All DeepScale documentation, tutorials, and blogs can be found on our website: [deepscale.khulnasoft.com](https://www.deepscale.khulnasoft.com/)


|                                                                                                | Description                                  |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [Getting Started](https://www.deepscale.khulnasoft.com/getting-started/)                                   |  First steps with DeepScale                  |
| [DeepScale JSON Configuration](https://www.deepscale.khulnasoft.com/docs/config-json/)                     |  Configuring DeepScale                       |
| [API Documentation](https://deepscale.readthedocs.io/en/latest/)                               |  Generated DeepScale API documentation       |
| [Tutorials](https://www.deepscale.khulnasoft.com/tutorials/)                                               |  Tutorials                                   |
| [Blogs](https://www.deepscale.khulnasoft.com/posts/)                                                       |  Blogs                                   |


# Contributing
DeepScale welcomes your contributions! Please see our
[contributing](CONTRIBUTING.md) guide for more details on formatting, testing,
etc.<br/>
Thanks so much to all of our amazing contributors!

<a href="https://github.com/khulnasoft/DeepScale/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=khulnasoft/DeepScale&r="  width="800px"/>
</a>
