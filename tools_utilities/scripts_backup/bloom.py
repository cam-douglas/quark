import torch

from gptq import (
    Quantizer,
    find_layers,
    get_loaders,
    quantize,
)
from gptq.utils import (
    get_model,
    Catcher,
    model_pack,
    load_quant,
    move_to_device,
    get_quantizer,
    get_perplexity,
    get_args, DATASET_LIST
)
import transformers
from transformers import BloomForCausalLM, BloomConfig


@torch.no_grad()
def eval_model(model, testenc, args, dev):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h
    layers_to_move = ["word_embeddings", "word_embeddings_layernorm"]
    move_to_device(model.transformer, layers_to_move, dev)

    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cat = Catcher(layers[0], inps)
    layers[0] = cat
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    move_to_device(model.transformer, layers_to_move, torch.device("cpu"))
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cat.cache["attention_mask"]
    alibi = cat.cache["alibi"]

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    logits = []
    for i in range(nsamples):
        hidden_states = model.transformer.ln_f(inps[i].unsqueeze(0))
        logits.append(model.lm_head(hidden_states))
    model.config.use_cache = use_cache
    return get_perplexity(logits, model.seqlen, testenc)


def run():
    args = get_args()
    if args.load:
        config = BloomConfig.from_pretrained(args.model)
        torch.set_default_dtype(torch.half)
        model = BloomForCausalLM(config)
        torch.set_default_dtype(torch.float)
        skip_layers = ["lm_head"]
        model = load_quant(
            model,
            args.load,
            args.wbits,
            skip_layers,
            seqlen=model.config.seq_length,
        )
    else:
        model = get_model(BloomForCausalLM, args.model)
        model.seqlen = model.config.seq_length
        # print(model)
        model.eval()
    dev = (
        torch.device(args.cuda) if args.cuda.startswith("cuda") else torch.device("cpu")
    )
    if args.eval:
        for dataset in DATASET_LIST:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            eval_model(model, testloader, args, dev)
        exit(0)

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    if args.wbits < 16 and not args.nearest:
        layers_to_move = ["word_embeddings", "word_embeddings_layernorm"]
        layer_kws = ["attention_mask", "alibi"]
        quantizers = get_quantizer(
            model,
            model.transformer,
            model.transformer.h,
            "transformer.h",
            dataloader,
            args,
            layers_to_move,
            layer_kws,
            dev,
        )

    if args.save:
        model_pack(model, quantizers, args.wbits)
        torch.save(model.state_dict(), args.save)


"""
A lower perplexity indicates a better model.
https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
# generate 4bit bloom
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-1b7 c4 --wbits 4 --save bloom-1.7B4b.pt

Result:
CUDA_VISIBLE_DEVICES=2 python bloom.py bigscience/bloom-1b7 --eval
wikitext2: 20.602636337280273
PTB: 41.62135314941406
C4: 21.91985511779785

CUDA_VISIBLE_DEVICES=2 python bloom.py bigscience/bloom-1b7 c4 --wbits 4 --load bloom-1.7B4b.pt --eval
wikitext2: 22.730281829833984
PTB: 45.70764923095703
C4: -
"""

if __name__ == "__main__":
    run()
