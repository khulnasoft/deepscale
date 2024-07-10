[![Build Status](https://github.com/khulnasoft/deepscale/workflows/Build/badge.svg)](https://github.com/khulnasoft/DeepScale/actions)
[![PyPI version](https://badge.fury.io/py/deepscale.svg)](https://pypi.org/project/deepscale/)
[![Documentation Status](https://readthedocs.org/projects/deepscale/badge/?version=latest)](https://deepscale.readthedocs.io/en/latest/?badge=latest)
[![License MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Khulnasoft/DeepScale/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/deepscale/month)](https://pepy.tech/project/deepscale)

### DeepScale is hiring! Come join us: [SDE 2](https://careers.khulnasoft.com/us/en/job/1013160/Software-Engineer-2), [Sr. SDE](https://careers.khulnasoft.com/us/en/job/1017151/Senior-Software-Engineer), [Sr. Researcher](https://careers.khulnasoft.com/us/en/job/1016440/Senior-Researcher)

[DeepScale](https://www.deepscale.khulnasoft.com/) is a deep learning optimization
library that makes distributed training easy, efficient, and effective.

<p align="center"><i><b>10x Larger Models</b></i></p>
<p align="center"><i><b>10x Faster Training</b></i></p>
<p align="center"><i><b>Minimal Code Change</b></i></p>

DeepScale delivers extreme-scale model training for everyone, from data scientists training on massive supercomputers to those training on low-end clusters or even on a single GPU:
* Extreme scale: Using current generation of GPU clusters with hundreds of devices,  3D parallelism of DeepScale can efficiently train deep learning models with trillions of parameters.  
* Extremely memory efficient: With just a single GPU, ZeRO-Offload of DeepScale can train models with over 10B parameters, 10x bigger than the state of arts, democratizing multi-billion-parameter model training such that many deep learning scientists can explore bigger and better models.
* Extremely long sequence length: Sparse attention of DeepScale powers an order-of-magnitude longer input sequence and obtains up to 6x faster execution comparing with dense transformers.  
* Extremely communication efficient: 3D parallelism improves communication efficiency allows users to train multi-billion-parameter models 2–7x faster on clusters with limited network bandwidth.  1-bit Adam/1-bit LAMB reduce communication volume by up to 5x while achieving similar convergence efficiency to Adam/LAMB, allowing for scaling to different types of GPU clusters and networks.

Early adopters of DeepScale have already produced
a language model (LM) with over 17B parameters called
[Turing-NLG](https://www.khulnasoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-khulnasoft),
establishing a new SOTA in the LM category.

DeepScale is an important part of Khulnasoft’s new
[AI at Scale](https://www.khulnasoft.com/en-us/research/project/ai-at-scale/)
initiative to enable next-generation AI capabilities at scale, where you can find more
information [here](https://innovation.khulnasoft.com/en-us/exploring-ai-at-scale).

**_For further documentation, tutorials, and technical deep-dives please see [deepscale.khulnasoft.com](https://www.deepscale.khulnasoft.com/)!_**


# News
* [2021/08/16] [Curriculum learning: a regularization method for stable and 2.6x faster GPT-2 pre-training with 8x/4x larger batch size/learning rate](https://www.deepscale.khulnasoft.com/tutorials/curriculum-learning/)
* [2021/05/24] [DeepScale: Accelerating large-scale model inference and training via system optimizations and compression](https://www.khulnasoft.com/en-us/research/blog/deepscale-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/)
* [2021/04/20] [1-bit LAMB: up to 4.6x less communication and 2.8x faster training, together with LAMB's convergence speed at large batch sizes](https://www.deepscale.khulnasoft.com/tutorials/onebit-lamb/)
* [2021/04/19] [ZeRO-Infinity unlocks unprecedented model scale for deep learning training](https://www.khulnasoft.com/en-us/research/blog/zero-infinity-and-deepscale-unlocking-unprecedented-model-scale-for-deep-learning-training/)
  * [Tutorial on how to use different stages of ZeRO](https://www.deepscale.khulnasoft.com/tutorials/zero/)
* [2021/04/01] [[DeepScale on AzureML] Transformers and CIFAR examples are now available on AzureML GitHub](https://github.com/Azure/azureml-examples/tree/main/python-sdk/workflows/train/deepscale)
* [2021/03/30] [[PyTorch Lightning Blog] Accessible Multi-Billion Parameter Model Training with PyTorch Lightning + DeepScale](https://medium.com/pytorch-lightning/accessible-multi-billion-parameter-model-training-with-pytorch-lightning-deepscale-c9333ac3bb59)
* [2021/03/16] [1-bit Adam v2: NCCL-based implementation and more](https://www.deepscale.khulnasoft.com/tutorials/onebit-adam/)
* [2021/03/08] [ZeRO-3 Offload: Scale your models to trillion parameters without code changes while leveraging both CPUs & GPUs](https://www.deepscale.khulnasoft.com/news/2021/03/07/zero3-offload.html)
* [2021/01/19] [[🤗Hugging Face Blog] Fit More and Train Faster With ZeRO via DeepScale and FairScale](https://huggingface.co/blog/zero-deepscale-fairscale)
* [2020/11/12] [Simplified install, JIT compiled ops, PyPI releases, and reduced dependencies](#installation)
* [2020/11/10] [Efficient and robust compressed training through progressive layer dropping](https://www.deepscale.khulnasoft.com/news/2020/10/28/progressive-layer-dropping-news.html)
* [2020/09/10] [DeepScale v0.3: Extreme-scale model training for everyone](https://www.khulnasoft.com/en-us/research/blog/deepscale-extreme-scale-model-training-for-everyone/)


# Table of Contents
| Section                                 | Description                                 |
| --------------------------------------- | ------------------------------------------- |
| [Why DeepScale?](#why-deepscale)        |  DeepScale overview                         |
| [Install](#installation)                |  Installation details                       |
| [Features](#features)                   |  Feature list and overview                  |
| [Further Reading](#further-reading)     |  Documentation, tutorials, etc.             |
| [Contributing](#contributing)           |  Instructions for contributing              |
| [Publications](#publications)           |  Publications related to DeepScale          |
| [Videos](#videos)                       |  Videos related to DeepScale                |

# Why DeepScale?
Training advanced deep learning models is challenging. Beyond model design,
model scientists also need to set up the state-of-the-art training techniques
such as distributed training, mixed precision, gradient accumulation, and
checkpointing. Yet still, scientists may not achieve the desired system
performance and convergence rate. Large model sizes are even more challenging:
a large model easily runs out of memory with pure data parallelism and it is
difficult to use model parallelism. DeepScale addresses these challenges to
accelerate model development *and* training.

# Installation

The quickest way to get started with DeepScale is via pip, this will install
the latest release of DeepScale which is not tied to specific PyTorch or CUDA
versions. DeepScale includes several C++/CUDA extensions that we commonly refer
to as our 'ops'.  By default, all of these extensions/ops will be built
just-in-time (JIT) using [torch's JIT C++ extension loader that relies on
ninja](https://pytorch.org/docs/stable/cpp_extension.html) to build and
dynamically link them at runtime.

**Note:** [PyTorch](https://pytorch.org/) must be installed _before_ installing
DeepScale.

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

On Windows you can build wheel with following steps, currently only inference mode is supported.
1. Install pytorch, such as pytorch 1.8 + cuda 11.1
2. Install visual cpp build tools, such as VS2019 C++ x64/x86 build tools
3. Launch cmd console with Administrator privilege for creating required symlink folders
4. Run `python setup.py bdist_wheel` to build wheel in `dist` folder

# Features
Below we provide a brief feature list, see our detailed [feature
overview](https://www.deepscale.khulnasoft.com/features/) for descriptions and usage.

* [Distributed Training with Mixed Precision](https://www.deepscale.khulnasoft.com/features/#distributed-training-with-mixed-precision)
  * 16-bit mixed precision
  * Single-GPU/Multi-GPU/Multi-Node
* [Model Parallelism](https://www.deepscale.khulnasoft.com/features/#model-parallelism)
  * Support for Custom Model Parallelism
  * Integration with Megatron-LM
* [Pipeline Parallelism](https://www.deepscale.khulnasoft.com/tutorials/pipeline/)
  * 3D Parallelism
* [The Zero Redundancy Optimizer (ZeRO)](https://www.deepscale.khulnasoft.com/tutorials/zero/)
  * Optimizer State and Gradient Partitioning
  * Activation Partitioning
  * Constant Buffer Optimization
  * Contiguous Memory Optimization
* [ZeRO-Offload](https://www.deepscale.khulnasoft.com/tutorials/zero-offload/)
  * Leverage both CPU/GPU memory for model training
  * Support 10B model training on a single GPU
* [Ultra-fast dense transformer kernels](https://www.deepscale.khulnasoft.com/news/2020/05/18/bert-record.html)
* [Sparse attention](https://www.deepscale.khulnasoft.com/news/2020/09/08/sparse-attention.html)
  * Memory- and compute-efficient sparse kernels
  * Support 10x longer sequences than dense
  * Flexible support to different sparse structures
* [1-bit Adam](https://www.deepscale.khulnasoft.com/news/2020/09/08/onebit-adam-blog-post.html) and [1-bit LAMB](https://www.deepscale.khulnasoft.com/tutorials/onebit-lamb/)
  * Custom communication collective
  * Up to 5x communication volume saving
* [Additional Memory and Bandwidth Optimizations](https://www.deepscale.khulnasoft.com/features/#additional-memory-and-bandwidth-optimizations)
  * Smart Gradient Accumulation
  * Communication/Computation Overlap
* [Training Features](https://www.deepscale.khulnasoft.com/features/#training-features)
  * Simplified training API
  * Gradient Clipping
  * Automatic loss scaling with mixed precision
* [Training Optimizers](https://www.deepscale.khulnasoft.com/features/#training-optimizers)
  * Fused Adam optimizer and arbitrary `torch.optim.Optimizer`
  * Memory bandwidth optimized FP16 Optimizer
  * Large Batch Training with LAMB Optimizer
  * Memory efficient Training with ZeRO Optimizer
  * CPU-Adam
* [Training Agnostic Checkpointing](https://www.deepscale.khulnasoft.com/features/#training-agnostic-checkpointing)
* [Advanced Parameter Search](https://www.deepscale.khulnasoft.com/features/#advanced-parameter-search)
  * Learning Rate Range Test
  * 1Cycle Learning Rate Schedule
* [Simplified Data Loader](https://www.deepscale.khulnasoft.com/features/#simplified-data-loader)
* [Curriculum Learning](https://www.deepscale.khulnasoft.com/tutorials/curriculum-learning/)
  * A curriculum learning-based data pipeline that presents easier or simpler examples earlier during training
  * Stable and 2.6x faster GPT-2 pre-training with 8x/4x larger batch size/learning rate while maintaining token-wise convergence speed
  * Complementary to many other DeepScale features
* [Performance Analysis and Debugging](https://www.deepscale.khulnasoft.com/features/#performance-analysis-and-debugging)



# Further Reading

All DeepScale documentation can be found on our website: [deepscale.khulnasoft.com](https://www.deepscale.khulnasoft.com/)


| Article                                                                                        | Description                                  |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [DeepScale Features](https://www.deepscale.khulnasoft.com/features/)                                       |  DeepScale features                          |
| [Getting Started](https://www.deepscale.khulnasoft.com/getting-started/)                                   |  First steps with DeepScale                  |
| [DeepScale JSON Configuration](https://www.deepscale.khulnasoft.com/docs/config-json/)                     |  Configuring DeepScale                       |
| [API Documentation](https://deepscale.readthedocs.io/en/latest/)                               |  Generated DeepScale API documentation       |
| [CIFAR-10 Tutorial](https://www.deepscale.khulnasoft.com/tutorials/cifar-10)                               |  Getting started with CIFAR-10 and DeepScale |
| [Megatron-LM Tutorial](https://www.deepscale.khulnasoft.com/tutorials/megatron/)                           |  Train GPT2 with DeepScale and Megatron-LM   |
| [BERT Pre-training Tutorial](https://www.deepscale.khulnasoft.com/tutorials/bert-pretraining/)             |  Pre-train BERT with DeepScale               |
| [Learning Rate Range Test Tutorial](https://www.deepscale.khulnasoft.com/tutorials/lrrt/)                  |  Faster training with large learning rates   |
| [1Cycle Tutorial](https://www.deepscale.khulnasoft.com/tutorials/one-cycle/)                               |  SOTA learning schedule in DeepScale         |



# Contributing
DeepScale welcomes your contributions! Please see our
[contributing](CONTRIBUTING.md) guide for more details on formatting, testing,
etc.

## Contributor License Agreement
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. For details, visit
https://cla.opensource.khulnasoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply
follow the instructions provided by the bot. You will only need to do this once across
all repos using our CLA.

## Code of Conduct
This project has adopted the [Khulnasoft Open Source Code of
Conduct](https://opensource.khulnasoft.com/codeofconduct/). For more information see the
[Code of Conduct FAQ](https://opensource.khulnasoft.com/codeofconduct/faq/) or contact
[opencode@khulnasoft.com](mailto:opencode@khulnasoft.com) with any additional questions or comments.
