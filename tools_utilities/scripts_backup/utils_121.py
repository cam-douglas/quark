import copy
import math
import time

import torch
import transformers

from .....................................................gptq import GPTQ
from .....................................................modelutils import find_layers
from .....................................................quant import Quantizer, make_quant, QuantLinear

DATASET_LIST = ["c4", "wikitext2", "ptb"]


def non_ops(*args, **kwargs):
    pass


def avoid_tensor_modified():
    torch.nn.init.kaiming_uniform_ = non_ops
    torch.nn.init.uniform_ = non_ops
    torch.nn.init.normal_ = non_ops


def get_model(transformer_model, model_name, seqlen=1024):
    avoid_tensor_modified()
    model = transformer_model.from_pretrained(model_name, torch_dtype="auto")
    model.seqlen = seqlen  # model.config.max_position_embeddings
    return model


class Catcher(torch.nn.Module):
    def __init__(self, module, inps):
        super().__init__()
        self.module = module
        self.cache = {"i": 0, "attention_mask": None}
        self.inps = inps

    def forward(self, inp, **kwargs):
        self.inps[self.cache["i"]] = inp
        self.cache["i"] += 1
        self.cache["attention_mask"] = kwargs["attention_mask"]
        # for bloom
        if "alibi" in kwargs:
            self.cache["alibi"] = kwargs["alibi"]
        raise ValueError


class Mover(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.dev = next(iter(self.module.parameters())).device
        self.cache = {"mask": None}

    def forward(self, *inp, **kwargs):
        inp = list(inp)
        if inp[0].device != self.dev:
            inp[0] = inp[0].to(self.dev)
        if self.cache["mask"] is None or self.cache["mask"].device != self.dev:
            self.cache["mask"] = kwargs["attention_mask"].to(self.dev)
        kwargs["attention_mask"] = self.cache["mask"]
        tmp = self.module(*inp, **kwargs)
        return tmp


def add_batch(gptq_):
    def func(_, inp, out):
        gptq_.add_batch(inp[0].data, out.data)

    return func


@torch.no_grad()
def get_quantizer(model, decoder, layers, layer_str, dataloader, args, layers_to_move, layer_kws, dev):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    move_to_device(decoder, layers_to_move, dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cat = Catcher(layers[0], inps)
    layers[0] = cat
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    move_to_device(decoder, layers_to_move, torch.device("cpu"))
    torch.cuda.empty_cache()
    layer_kwargs = {i: cat.cache[i] for i in layer_kws}
    quantizers = quantize_layers(
        layers,
        cat.inps,
        layer_kwargs,
        args.wbits,
        args.nsamples,
        args.percdamp,
        args.groupsize,
        dev,
        layer_str,
    )
    model.config.use_cache = use_cache
    return quantizers


def quantize_layers(layers, inps, layer_kwargs, wbits, nsamples, percdamp, groupsize, dev, layer_str, perchannel=True,
                    sym=False, mse=False):
    quantizers = {}
    outs = torch.zeros_like(inps)
    for i, layer in enumerate(layers):
        layer = layer.to(dev)
        name_to_module = find_layers(layer)
        name_to_gptq = {
            name: GPTQ(module) for name, module in name_to_module.items()
        }
        for gptq in name_to_gptq.values():
            gptq.quantizer = Quantizer()
            gptq.quantizer.configure(wbits, perchannel=perchannel, sym=sym, mse=mse)
        handles = [
            module.register_forward_hook(add_batch(name_to_gptq[name]))
            for name, module in name_to_module.items()
        ]
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()
        print(f"\nQuantize layer: {i}", end=",")
        for name, gptq in name_to_gptq.items():
            print(name, end=",")
            gptq.fasterquant(percdamp=percdamp, groupsize=groupsize)
            quantizers[f"{layer_str}.{i}.{name}"] = gptq.quantizer
            gptq.free()
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer, name_to_gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    return quantizers


# TODO: perform packing on GPU
def model_pack(
        model, quantizers, wbits, dev=torch.device("cpu"), quant_layers=[QuantLinear]
):
    layers = find_layers(model)
    layers = {nm: layers[nm] for nm in quantizers}
    make_quant(model, quantizers, wbits)
    qlayers = find_layers(model, quant_layers)
    for name, qlayer in qlayers.items():
        quantizer = quantizers[name].to(dev)
        qlayer.pack(layers[name], quantizer.scale, quantizer.zero)
    return model


# ["model.decoder.project_out", "model.decoder.project_in", "lm_head"]
def load_quant(
        model, checkpoint, wbits, skip_layers=[], seqlen=1024, for_infer=True, verbose=False
):
    avoid_tensor_modified()
    transformers.modeling_utils._init_weights = False
    if for_infer:
        model.eval()
    layers = find_layers(model)
    layers = {n: layers[n] for n in layers if n not in skip_layers}
    make_quant(model, layers, wbits)
    if verbose:
        print(f"⌛️ Loading model from {checkpoint}...")
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = seqlen
    if verbose:
        print(f"✅ Model from {checkpoint} is loaded successfully.")

    return model


def move_to_device(l, attrs, dev):
    [
        setattr(l, e, getattr(l, e).to(dev))
        for e in attrs
        if hasattr(l, e) and getattr(l, e)
    ]


def decoder_multigpu(model, decoder, gpus, first=[], last=[]):
    # Move specified modules to first/last GPU
    move_to_device(decoder, first, gpus[0])
    move_to_device(decoder, last, gpus[-1])
    # Deep copy the model's LM head and move it to the last GPU
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])
    # Split the decoder layers across GPUs
    per_gpu = math.ceil(len(decoder.layers) / len(gpus))
    decoder.layers = [
        Mover(layer.to(gpus[i // per_gpu])) for i, layer in enumerate(decoder.layers)
    ]
    # Set the model's GPUs
    model.gpus = gpus


def sync_model(model):
    if hasattr(model, "gpus"):
        for gpu in model.gpus:
            torch.cuda.synchronize(gpu)
    else:
        torch.cuda.synchronize()


def benchmark(model, decoder, input_ids, check=False, dev=torch.device("cuda:0")):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else dev)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = torch.nn.CrossEntropyLoss()
        tot = 0.0

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=dev)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape(-1),
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync_model(model)
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(
                    out.logits[0].to(dev), input_ids[:, (i + 1)].to(dev)
                ).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync_model(model)
        import numpy as np

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
            print("max memory(MiB):", max_memory)


def get_perplexity(logits, seqlen, testenc):
    nlls = []
    nsamples = len(logits)
    for i, lm_logits in enumerate(logits):
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    return torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="pretrained model to load")
    parser.add_argument(
        "dataset",
        type=str,
        default="c4",
        const="c4",
        nargs="?",
        choices=DATASET_LIST,
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--nearest", action="store_true", help="Whether to run the RTN baseline."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=8,
        choices=[2, 3, 4, 8],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save quantized checkpoint under this name.",
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Number of tokens to use for benchmarking.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to compute perplexity during benchmarking for verification.",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="cuda:0",
        help="GPU device string, 'cuda:0' by default.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the model with dataset wikitext2, ptb and c4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(Catcher.cache)
