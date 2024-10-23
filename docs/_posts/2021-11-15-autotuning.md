---
title: "Autotuning: Automatically discover the optimal DeepScale configuration that delivers good training speed"
excerpt: ""
date: 2021-11-16 10:00:00
tags: training English
toc: false
---

We introduce a new feature called Autotuning to automatically discover the optimal DeepScale configuration that delivers good training speed. One pain point in model training is to figure out good performance-relevant configurations such as micro-batch size to fully utilize the hardware and achieve a high throughput number. This configuration exploring process is commonly done manually but is important since model training is repeated many times and benefits from using a good configuration. Not only is the hand-tuning process time-consuming, but the outcome is hardware-dependent. This means that a good configuration on one hardware might not be the best on another different hardware. The user thus has to hand tune the configuration again. With DeepScale, there are more configuration parameters that could potentially affect the training speed, thus making it more tedious to manually tune the configuration.

The DeepScale Autotuner mitigates this pain point and automatically discovers the optimal DeepScale configuration that delivers good training speed. It not only reduces the time and resources users spend on tuning, but also can discover configurations better than hand-tuned methods. [DeepScaleExamples](https://github.com/khulnasoft/DeepScaleExamples/tree/master/autotuning) would demonstrate the effectiveness of autotuning across different models.

* For a brief overview, see the [Autotuning tutorial](https://www.deepscale.ai/tutorials/autotuning/).
* For more information on how to use Autotuning, see the [Autotuning README](https://github.com/khulnasoft/DeepScale/tree/master/deepscale/autotuning#deepscale-autotuning).
* The source code can be found in the [DeepScale repo](https://github.com/khulnasoft/deepscale).
