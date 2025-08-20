import numpy as np
import torch
from quant_cuda import matvmul2, matvmul3, matvmul4, matvmul8


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(torch.nn.Module):
    """Round to nearest quantizer"""
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=0.8):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(torch.nn.Module):
    """GPTQ quantizer"""
    def __init__(self, bits, infeatures, outfeatures):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.bits = bits
        self.register_buffer("zeros", torch.zeros((outfeatures, 1)))
        self.register_buffer("scales", torch.zeros((outfeatures, 1)))
        self.register_buffer("bias", torch.zeros(outfeatures))
        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 256 * (bits * 8), outfeatures), dtype=torch.int),
        )

    def pack(self, linear, scales, zeros):
        """Pack the weights of a given linear layer into a quantized format.

        The method takes in three arguments: linear, which is a PyTorch linear layer whose weights will be
        quantized; scales, which is a scaling factor used for quantization; and zeros, which is an offset
        factor used for quantization.

        First, the method computes the zeros offset by multiplying zeros and scales. It then creates a clone
        of the scales tensor and, if the linear layer has a bias tensor, creates a clone of the bias tensor as well.

        Next, the method quantizes the weights of the linear layer. It rounds the weights to the nearest
        integer, adds the zeros offset, and divides by the scales factor. The resulting quantized weights
        are stored as a 2D numpy array called qweight, where the rows correspond to different quantization
        chunks and the columns correspond to different weights in the linear layer.

        The quantization method depends on the number of bits specified by the self.bits parameter.
        If self.bits is 2, 4, 8, or 16, the quantization method packs multiple weights into a single 32-bit word using bit-shifting operations.
        If self.bits is 3, the quantization method uses a more complex packing method that packs 10 weights into 32 bits using a combination of bit-shifting and masking operations.

        Finally, the qweight numpy array is converted to a PyTorch tensor and stored as a module buffer called self.qweight.
        Overall, this pack method is used to perform quantization of weights for a given linear layer in a neural network.
        """
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(
            torch.int
        )
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 256 * (self.bits * 8), intweight.shape[1]),
            dtype=np.uint32,
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8, 16]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        outshape = list(x.shape)
        x = x.reshape(-1, x.shape[-1])
        y = self.bias.clone().repeat(x.shape[0], 1)
        outshape[-1] = self.bias.numel()
        dtype = x.dtype
        x = x.float()
        if self.bits == 2:
            matvmul2(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 3:
            matvmul3(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 4:
            matvmul4(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 8:
            matvmul8(x, self.qweight, y, self.scales, self.zeros)
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        y = y.to(dtype)
        return y.reshape(outshape)


def make_quant(module, names, bits, name=""):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        nm = name + "." + attr if name != "" else attr
        if nm in names:
            setattr(module, attr, QuantLinear(bits, tmp.in_features, tmp.out_features))
    for nm, child in module.named_children():
        make_quant(child, names, bits, name + "." + nm if name != "" else nm)


if __name__ == "__main__":
    pass
