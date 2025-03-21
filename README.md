# Extreme Speed and Scale for DL Training and Inference

***[DeepScale](https://www.deepscale.khulnasoft.com/) enables world's most powerful language models like [MT-530B](https://www.microsoft.com/en-us/research/blog/using-deepscale-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) and [BLOOM](https://huggingface.co/blog/bloom-megatron-deepscale)***. It is an easy-to-use deep learning optimization software suite that powers unprecedented scale and speed for both training and inference. With DeepScale you can:

* Train/Inference dense or sparse models with billions or trillions of parameters
* Achieve excellent system throughput and efficiently scale to thousands of GPUs
* Train/Inference on resource constrained GPU systems
* Achieve unprecedented low latency and high throughput for inference
* Achieve extreme compression for an unparalleled inference latency and model size reduction with low costs

---

[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Microsoft/DeepScale/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/deepscale.svg)](https://pypi.org/project/deepscale/)
[![Downloads](https://static.pepy.tech/badge/deepscale)](https://pepy.tech/project/deepscale)
[![Build](https://badgen.net/badge/build/check-status/blue)](#build-pipeline-status)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9530/badge)](https://www.bestpractices.dev/projects/9530)
[![Twitter](https://img.shields.io/twitter/follow/MSFTDeepScale)](https://twitter.com/intent/follow?screen_name=MSFTDeepScale)
[![Japanese Twitter](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9ETwitter-%40MSFTDeepScaleJP-blue)](https://twitter.com/MSFTDeepScaleJP)
[![Chinese Zhihu](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%BE%AE%E8%BD%AFDeepScale-blue)](https://www.zhihu.com/people/deepscale)

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

