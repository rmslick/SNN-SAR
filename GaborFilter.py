import SpykeTorch.utils as utils

kernels = [	utils.GaborKernel(window_size = 3, orientation = 45+22.5),
            utils.GaborKernel(3, 90+22.5),
            utils.GaborKernel(3, 135+22.5),
            utils.GaborKernel(3, 180+22.5)]
filter = utils.Filter(kernels, use_abs = True)