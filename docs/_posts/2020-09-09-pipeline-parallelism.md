---
layout: single
title: "Training a Trillion Parameters with Pipeline Parallelism"
excerpt: ""
categories: news
new_post: true
date: 2020-09-09 00:00:00
---

DeepScale includes new support for pipeline parallelism! DeepScale's training
engine provides hybrid 3D parallelism for training models with over a
trillion parameters. In addition to scaling to the extreme, we have
demonstrated that hybrid parallelism accelerates training on clusters with
low-bandwidth network by up to 7x.

* For a brief overview and results including trillion-parameter capabilities,
  see our [press release]({{ site.press_release_v3 }}).
* To get started with pipeline parallel training in DeepScale, we recommend our [tutorial](/tutorials/pipeline/).
* See our AlexNet example in [DeepScaleExamples](https://github.com/khulnasoft-lab/DeepScaleExamples).
* Read our API documentation on [readthedocs](https://deepscale.readthedocs.io/en/latest/pipeline.html).
