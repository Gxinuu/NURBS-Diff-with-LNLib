/*
 * Author:
 * Copyright (c) 2025/02/24- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 *
 */

#pragma once

#include "ND_Definitions.h"
#include <torch/torch.h>

using namespace torch::autograd;

namespace ND_LNLib
{
	class ND_LNLib_EXPORT CurveEvalFunction:public Function<CurveEvalFunction>
	{

	public:
		static void forward(AutogradContext* ctx);
		static void backward(AutogradContext* ctx);

	};

}



