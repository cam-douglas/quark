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
    benchmark,
    move_to_device,
    decoder_multigpu,
    get_quantizer,
    get_args,
    get_perplexity,DATASET_LIST
)
from transformers import OPTForCausalLM, OPTConfig


@torch.no_grad()
def eval_model(model, testenc, args, dev):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    decoder = model.model.decoder
    layers = decoder.layers
    layers_to_move = ["embed_tokens", "embed_positions", "project_out", "project_in"]
    move_to_device(decoder, layers_to_move, dev)
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
    inps = cat.inps
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    move_to_device(decoder, layers_to_move, torch.device("cpu"))
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cat.cache["attention_mask"]
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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    move_to_device(decoder, ["final_layer_norm", "project_out"], dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    logits = []
    # Perplexity as the exponential of the cross-entropy
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if decoder.final_layer_norm is not None:
            hidden_states = decoder.final_layer_norm(hidden_states)
        if decoder.project_out is not None:
            hidden_states = decoder.project_out(hidden_states)
        logits.append(model.lm_head(hidden_states))
    model.config.use_cache = use_cache
    return get_perplexity(logits, model.seqlen, testenc)


def run():
    args = get_args()
    if args.load:
        config = OPTConfig.from_pretrained(args.model)
        torch.set_default_dtype(torch.half)
        model = OPTForCausalLM(config)
        torch.set_default_dtype(torch.float)
        skip_layers = [
            "model.decoder.project_out",
            "model.decoder.project_in",
            "lm_head",
        ]
        model = load_quant(
            model,
            args.load,
            args.wbits,
            skip_layers,
            model.config.max_position_embeddings,
        )
    else:
        model = get_model(OPTForCausalLM, args.model, 2048)
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
            ppl = eval_model(model, testloader, args, dev)
            print(ppl)
        exit(0)

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if args.wbits < 16 and not args.nearest:
        layers_to_move = [
            "embed_tokens",
            "embed_positions",
            "project_out",
            "project_in",
        ]
        layer_kws = ["attention_mask"]
        quantizers = get_quantizer(
            model,
            model.model.decoder,
            model.model.decoder.layers,
            "model.decoder.layers",
            dataloader,
            args,
            layers_to_move,
            layer_kws,
            dev,
        )

    if args.benchmark:
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            first = ["embed_tokens", "embed_positions", "project_in"]
            last = ["project_out", "final_layer_norm"]
            decoder_multigpu(model, model.model.decoder, gpus, first, last)
        else:
            model = model.to(dev)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, : args.benchmark]
            benchmark(model, model.model.decoder, input_ids, check=args.check)

    if args.save:
        model_pack(model, quantizers, args.wbits)
        torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    run()
