---
layout: single
title: "The Fastest and Most Efficient BERT Training through Optimized Transformer Kernels"
excerpt: ""
categories: news
new_post: true
date: 2020-05-19 00:00:00
---

We introduce new technology to accelerate single GPU performance via kernel
optimizations. These optimizations not only create a strong foundation for
scaling out large models, but also improve the single GPU performance of
highly tuned and moderately sized models like BERT by more than 30%, reaching
a staggering performance of 66 teraflops per V100 GPU, which is 52% of the
hardware peak. **Using optimized transformer kernels as the building block,
DeepScale achieves the fastest BERT training record: 44 minutes on 1,024
NVIDIA V100 GPUs**, compared with the best published result of 67 minutes on
the same number and generation of GPUs.

* Brief overview, see our [press release](https://www.khulnasoft.com/en-us/research/blog/zero-2-deepscale-shattering-barriers-of-deep-learning-speed-scale/).
* Detailed technology deep dive, see our [blog post](https://www.deepscale.khulnasoft.com/news/2020/05/27/fastest-bert-training.html).
* Tutorial on how to reproduce our results, see our [BERT pre-training tutorial](https://www.deepscale.khulnasoft.com/tutorials/bert-pretraining/).
* The source code for our transformer kernels can be found in the [DeepScale repo](https://github.com/khulnasoft/deepscale) and BERT pre-training code can be found in the [DeepScaleExamples repo](https://github.com/khulnasoft-lab/deepscaleexamples).
