---
layout: single
title: "Up to 5x less communication and 3.4x faster training through 1-bit Adam"
excerpt: ""
categories: news
new_post: true
date: 2020-09-09 00:00:00
---


Adam is an effective and probably the most well-utilized optimizer for
training many large-scale deep learning models.  However, Adam is generally
not compatible with communication-efficient optimization algorithms, and
therefore the communication cost could become a bottleneck while scaling
across distributed devices. We introduce a new algorithm - 1-bit Adam - and
its efficient implementation in DeepScale. 1-bit Adam offers the ***same convergence*** as Adam, incurs up to ***5x less communication*** that enables up to ***3.5x higher throughput for BERT-Large pretraining*** and up to ***2.7x higher throughput for SQuAD fine-tuning*** on bandwidth-limited clusters.

* Brief overview, see our [press release]({{ site.press_release_v3 }}).
* Detailed technology deep dive, see our [blog post](https://www.deepscale.khulnasoft.com/news/2020/09/08/onebit-adam-blog-post.html).
* Tutorial on how to reproduce our results, see our [1-bit Adam tutorial](/tutorials/onebit-adam/).
* The source code for 1-bit Adam can be found in the [DeepScale repo](https://github.com/khulnasoft/deepscale). The implementation of 1-bit Adam is in [onebit_adam.py](https://github.com/khulnasoft/DeepScale/blob/master/deepscale/runtime/fp16/onebit_adam.py) and CUDA-Aware communication for 1-bit Adam is in [custom_collectives.py](https://github.com/khulnasoft/DeepScale/blob/master/deepscale/runtime/custom_collectives.py). Example codes to try this feature can be found in the [DeepScaleExamples repo](https://github.com/khulnasoft-lab/deepscaleexamples) as shown in the [tutorial](/tutorials/onebit-adam/).