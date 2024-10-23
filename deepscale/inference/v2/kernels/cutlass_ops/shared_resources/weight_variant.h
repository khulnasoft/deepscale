// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

// Data structure that allows us to abstract internal CUTLASS datatypes/mappings
// to the DeepScale-Kernels repo.

#pragma once

enum WeightVariant { kFP16, kBF16, kFP8, kFP4 };
