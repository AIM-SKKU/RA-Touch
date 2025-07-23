import json
import os

from tqdm import tqdm
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from tvl_enc.tvl import TVL, ModalityType
from ra_touch.util.misc import download
from ra_touch.llama.tokenizer import Tokenizer
from ra_touch.llama.utils import sample_top_p
from ra_touch.llama.llama import Transformer, ModelArgs, RMSNorm
from ra_touch.tg_retriever import TGRetriever
from ra_touch.ta_integrator import TAIntegrator


class LLaMA_adapter(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, llama_ckpt_dir, llama_tokenizer, legacy_bridge=False, args=None, load=True):
        super().__init__()
        self.active_modality_names = args.active_modality_names

        feature_dim = 768
        self.max_words = args.max_words
        self.image_bind = TVL(active_modalities=[ModalityType.VISION, ModalityType.TACTILE], 
                              tactile_model=args.tactile_model)

        if args.checkpoint_path is not None:
            state_dict = torch.load(args.checkpoint_path, map_location='cpu')['model']
            miss_keys, unexpected_keys = self.image_bind.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {miss_keys}, unexpected keys: {unexpected_keys}")
                
        layers = [nn.Linear(feature_dim, 4096)]
        self.image_bind_proj = nn.Sequential(*layers)

        if legacy_bridge:
            bridge_norm_layer = nn.LayerNorm
            bridge_bias = True
        else:
            bridge_norm_layer = RMSNorm
            bridge_bias = False


        self.image_bind_norm_1 = bridge_norm_layer(4096)
        self.image_bind_f1_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.image_bind_f2_1 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.image_bind_f3_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.image_bind_norm_2 = bridge_norm_layer(4096)
        self.image_bind_f1_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.image_bind_f2_2 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.image_bind_f3_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.image_bind_norm_3 = bridge_norm_layer(4096)
        self.image_bind_f1_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.image_bind_f2_3 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.image_bind_f3_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        # 2. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 3. llama
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        phase = args.phase
        bias_lora = phase in ("finetune", "retrieval")
        model_args: ModelArgs = ModelArgs(
            max_seq_len=512, max_batch_size=1, w_bias=bias_lora, w_lora=bias_lora,
            **params
        ) # max_batch_size only affects inference
        print(f"model args: {model_args}")
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        # we enforce loading to llama checkpoint
        # if load: # this is deprecated
        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in tqdm(ckpts, desc="Loading LLaMA ckpt"):
            ckpt = torch.load(ckpt, map_location='cpu')
            names = self.llama.state_dict().keys()
            ckpt_names = ckpt.keys()
            for n in ckpt_names:
                if n not in names:
                    print(f"Warning: {n} not in llama model")
            self.llama.load_state_dict(ckpt, strict=False)
        self.llama_keys = ["llama." + i for i in ckpt_names]

        # 4. prefix
        self.query_layer = 32
        self.query_len = 1
        self.prefix_query = nn.Embedding(self.query_layer * self.query_len, model_args.dim)

        # 5. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.phase = phase
        
        # 6. retriever
        if self.phase in "retrieval":
            self.retrieval_method = args.retrieval_method
            self.retriever = TGRetriever(embed_path=args.embedding_path, 
                                         retrieval_method=self.retrieval_method)
            self.retriever.load_state_dict(torch.load(args.retriever_weight)['model'], strict=False)
            self.retriever.eval()
            print(f"Tactile-Guided Retriever loaded from {args.retriever_weight}")

            self.integrator = TAIntegrator(feature_dim=feature_dim, output_dim=4096)
            self.top_k = args.top_k
            
        self.set_default_trainability(self.phase)
        
    def get_trainable_params(self, phase='finetune'):
        trainable = {}
        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        trainable[name] = para
        elif phase == 'pretrain':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'gate' in name:
                        trainable[name] = para
                elif name.startswith("image_bind_"):  # not 'image_bind.' so image_bind won't be trained.
                    trainable[name] = para
                elif name.startswith("prefix_query."):
                    trainable[name] = para
        elif phase == 'retrieval':
            for name, para in self.named_parameters():
                if name.startswith("integrator"):
                    trainable[name] = para
                elif name.startswith("llama.layers"):
                    if 'norm' in name:
                        trainable[name] = para
                elif name.startswith("llama.norm"):
                    trainable[name] = para
        else:
            raise ValueError(f"Unknown model phase: {phase}")
        return trainable

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(LLaMA_adapter, self).state_dict(destination, prefix, keep_vars)
        # we remove all clip related weights and only save the tactile encoder
        new_state_dict = OrderedDict()
        for k in state_dict:
            param = state_dict[k]
            if "image_bind." in k or k in self.llama_keys: # the . is necessary 
                continue
            new_state_dict[k] = param
        del state_dict
        return new_state_dict

    def set_default_trainability(self, phase='finetune'):
        for key, value in self.named_parameters():
            value.requires_grad = False
        for key, value in self.get_trainable_params(phase).items():
            value.data = value.data.float()
            value.requires_grad = True

    def retrieve(self, query_emb: Tensor) -> Tuple[Tensor]:
        outs = self.retriever.retrieve(query_emb, top_k=self.top_k)
        img_embds, txt_embds = outs
        return img_embds, txt_embds
        
    def _get_retrieval_embeds(self, feats: Tensor, mode='train') -> Tensor:
        if mode == 'inference':
            self.integrator.eval()

        if feats[0][1] == 'vision':
            vis_feat = feats[0][0]
            tac_feat = feats[1][0]
        else:
            vis_feat = feats[1][0]
            tac_feat = feats[0][0]
        
        img_list, txt_list = [], []
        qry_embd = self.retriever(vis_feat=vis_feat,
                                  tac_feat=tac_feat)
        qry_embd = qry_embd['query_emb']
        
        for i in range(qry_embd.shape[0]):
            img, txt = self.retrieve(qry_embd[i][None])
            img_list.append(img)
            txt_list.append(txt)
        
        img_embeds = torch.stack(img_list, dim=0)
        txt_embeds = torch.stack(txt_list, dim=0)
        retrieval_feats = self.integrator(img_embeds, txt_embeds, tac_feat.unsqueeze(1))
        if retrieval_feats.dim() == 3:
            retrieval_feats = retrieval_feats.squeeze(1)
        
        return retrieval_feats

    def forward_visual(self, inputs, cache_size=10, cache_t=20, cache_weight=0.5):
        outputs = []
        outputs_weights = []
        for input_type, (input, input_weight) in inputs.items():
            type = input_type.lower()
            outputs.append([self.image_bind({type : input})[type], type])
            outputs_weights.append(input_weight)
        outputs_weights = [x/(sum(outputs_weights)+1e-6) for x in outputs_weights]


        visual_feats = sum([F.normalize(output[0], dim=-1) * output_weight 
                            for output, output_weight in zip(outputs, outputs_weights)])
        device = visual_feats.device

        visual_feats = visual_feats.unsqueeze(1) # B, 1, D
        visual_feats = self.image_bind_proj(visual_feats)
        visual_feats_norm = self.image_bind_norm_1(visual_feats)
        visual_feats = visual_feats + self.image_bind_f2_1(F.silu(self.image_bind_f1_1(visual_feats_norm)) * self.image_bind_f3_1(visual_feats_norm))

        visual_feats_norm = self.image_bind_norm_2(visual_feats)
        visual_feats = visual_feats + self.image_bind_f2_2(F.silu(self.image_bind_f1_2(visual_feats_norm)) * self.image_bind_f3_2(visual_feats_norm))

        visual_feats_norm = self.image_bind_norm_3(visual_feats)
        visual_feats = visual_feats + self.image_bind_f2_3(F.silu(self.image_bind_f1_3(visual_feats_norm)) * self.image_bind_f3_3(visual_feats_norm))
        return visual_feats, outputs
    
    @torch.inference_mode()
    def forward_inference(self, visual_feats, tokens, start_pos: int, retrieval_feats: Tensor = None):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)
        prefix_query = self.prefix_query.weight.reshape(
            self.query_layer, 1, 4096).unsqueeze(1)
        prefix_index = 0
        visual_proj = visual_feats # B, 1, D
        if retrieval_feats is not None:
            visual_proj = sum([visual_proj, retrieval_feats])

        for layer in self.llama.layers[-1 * self.query_layer:]:
            h = layer(h, start_pos, freqs_cis, mask, visual_proj + prefix_query[prefix_index].repeat(_bsz, 1, 1))
            prefix_index = prefix_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    def forward(self, tokens, labels, observations):
        imagebind_inputs = dict()
        for modality_name in self.active_modality_names:
            imagebind_inputs[modality_name] = [observations[modality_name], 1]
        visual_feats, feats = self.forward_visual(imagebind_inputs)
        _bsz, seqlen = tokens.shape
        
        if tokens.device != visual_feats.device:
            tokens = tokens.to(visual_feats.device)
            labels = labels.to(visual_feats.device)
        
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)
        prefix_query = self.prefix_query.weight.reshape(self.query_layer, 1, 4096).unsqueeze(1)
        prefix_index = 0
            
        visual_proj = visual_feats
        if self.phase == "retrieval":
            retrieval_feats = self._get_retrieval_embeds(feats)
            visual_proj = sum([visual_proj, retrieval_feats])
            
        for layer in self.llama.layers[-1 * self.query_layer:]:
            h = layer(h, 0, freqs_cis, mask, visual_proj + prefix_query[prefix_index])
            prefix_index = prefix_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        return c_loss, c_loss

    @torch.inference_mode()
    def generate(
            self,
            inputs,
            prompts,
            max_gen_len: int = 256,
            temperature: float = 0.1,
            top_p: float = 0.75,
            cache_size=10,
            cache_t=20,
            cache_weight=0.5
    ):
        bsz = len(prompts)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        with torch.cuda.amp.autocast():
            visual_query, feats = self.forward_visual(inputs, cache_size, cache_t, cache_weight)

        if isinstance(prompts[0], str):
            texts = [x for x in prompts]
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        
        if self.phase == "retrieval":
            retrieval_feats = self._get_retrieval_embeds(feats, mode='inference')
        else:
            retrieval_feats = None
        
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos, retrieval_feats)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded, texts


_MODELS = {
    "7B": "https://huggingface.co/Cxxs/ImageBind-LLM/resolve/main/7B.pth",
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts',
         llama_type="7B", args=None):
    if name in _MODELS:
        model_path = download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}")

    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    adapter_ckpt = torch.load(model_path, map_location='cpu')

    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path, args=args, load=True)

    adapter_ckpt["model"] = {k.replace("point_trunk", "modality_trunks.point") : v for k, v in adapter_ckpt["model"].items()}
    load_result = model.load_state_dict(adapter_ckpt['model'], strict=False)
    return model.to(device)