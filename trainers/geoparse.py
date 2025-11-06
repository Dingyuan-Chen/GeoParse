import os.path as osp
from collections import OrderedDict
import math
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from tqdm import tqdm
import os
from num2words import num2words

_tokenizer = _Tokenizer()


num_classes = 6
hiera = {
    "hiera_high":{
        "Greenhouse1":[0, 3],
        "Greenhouse2":[3, 5],
        "Greenhouse3":[5, 6],
    }
}
def prepare_targets(targets):
    c = targets.shape
    targets_high = torch.ones((c), dtype=targets.dtype, device=targets.device) * -100
    indices_high = []
    for index, high in enumerate(hiera["hiera_high"].keys()):
        indices = hiera["hiera_high"][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets == ii] = index
        indices_high.append(indices)

    return targets, targets_high, indices_high

def return_layout(fname='test_geometric concept.txt'):
    object_struct = {}
    with open(os.path.join('./data/', fname), 'r') as f:
        lines = f.readlines()
    for line in lines:
        line.strip()
        filename, cls, layouts = line.split(', ')
        span = float(layouts.split()[0])
        ratio = float(layouts.split()[1]) * 100.
        orientation = float(layouts.split()[2]) * 180. / np.pi
        object_struct[filename] = [span, ratio, orientation]
    return object_struct

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'GeoParse',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "geoparse_length": cfg.TRAINER.GEOPARSE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class GeoParseLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GEOPARSE.N_CTX
        ctx_init = cfg.TRAINER.GEOPARSE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.GEOPARSE.PROMPT_DEPTH >= 1, "For GEOPARSE, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.GEOPARSE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()

        self.layout_proj = nn.Linear(768, ctx_dim)
        self.layout_proj.half()
        self.layout_proj2 = nn.Linear(768, ctx_dim)
        self.layout_proj2.half()


        self.ctx = nn.Parameter(ctx_vectors)
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        classnames = [name.replace("_", " ") for name in classnames]

        attl = [
            ', the span is tiny',
            ', the span is small',
            ', the span is medium',
            ', the span is large',
            ', the span is enormous',
            ', the span is huge',
        ]
        # Secondary category
        classnames = [name + att for (name, att) in zip(classnames, attl)]
        # Primary category
        classnames.append('plastic tunnel, the orientation is large')
        classnames.append('Chinese solar greenhouse, the orientation is small')
        classnames.append('gutter connected greenhouse, the orientation is X')

        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls + 3, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = GeoParseLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.clip_model = clip_model
        self.rank_loss = nn.MarginRankingLoss()

    def layout_loss(self, logits, labels):
        losses = torch.zeros(1).cuda()
        y1 = torch.ones(1).cuda()

        if torch.unique(labels).size()[0] > 1:
            values, indexs = torch.sort(labels, descending=True)
            for ii in range(indexs.size()[0]):
                for jj in range(ii + 1, indexs.size()[0]):
                    if values[ii] > values[jj]:
                        losses += self.rank_loss(logits[indexs[ii]][values[ii]].unsqueeze(0), logits[indexs[jj]][values[ii]].unsqueeze(0), y1)

        return losses

    def forward(self, image, label=None, fnames=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

        if self.prompt_learner.training:
            object_struct = return_layout('train_geometric concept.txt')
        else:
            object_struct = return_layout('test_geometric concept.txt')

        ctx_lst = []
        for name in fnames:
            try:
                span = object_struct[os.path.basename(name)][0]
                orientation = object_struct[os.path.basename(name)][2]

                ctx_init_layout = 'the span is {}'.format(num2words(int(span)).replace('-', ' '))
                if len(ctx_init_layout.split()) > 4:
                    ctx_init_layout = ctx_init_layout.split()[0] + ' ' + ctx_init_layout.split()[1] + ' ' + ctx_init_layout.split()[2] + ' ' + ctx_init_layout.split()[3]

                ctx_init_ori = 'the orientation is {}'.format(num2words(int(orientation)).replace('-', ' '))
                if len(ctx_init_ori.split()) > 4:
                    ctx_init_ori = ctx_init_ori.split()[0] + ' ' + ctx_init_ori.split()[1] + ' ' + ctx_init_ori.split()[2] + ' ' + ctx_init_ori.split()[3]

                ctx_init = ctx_init_layout + ', ' + ctx_init_ori
            except:
                ctx_init = 'the span is X, the orientation is X'
                print('the span is X, the orientation is X')

            ctx_lst.append(ctx_init)

        layout_fea_lst = []
        layout_fea_lst.append(shared_ctx.expand(len(ctx_lst), -1, -1))

        span_fea_lst = []
        orien_fea_lst = []

        for ctx_init in ctx_lst:
            # ===================== hierarchy =====================
            ctx_init1, ctx_init2 = ctx_init.split(', ')
            span_ctx_init = ctx_init1.split()[0] + ' ' + ctx_init1.split()[1] + ' ' + \
                            ctx_init1.split()[2] + ' ' + ctx_init1.split()[3]
            orien_ctx_init = ctx_init2.split()[0] + ' ' + ctx_init2.split()[1] + ' ' + \
                            ctx_init2.split()[2] + ' ' + ctx_init2.split()[3]

            sprompt = clip.tokenize(span_ctx_init)
            sprompt = sprompt.to('cuda:0')
            oprompt = clip.tokenize(orien_ctx_init)
            oprompt = oprompt.to('cuda:0')

            n_hierar = 4

            with torch.no_grad():
                span_embedding = self.clip_model.token_embedding(sprompt).type(self.dtype)
            span_ctx_vectors = span_embedding[0, 1: 1 + n_hierar, :]
            span_ctx = nn.Parameter(span_ctx_vectors)
            span_prefix = span_embedding[:, :1, :]
            span_suffix = span_embedding[:, 1 + n_hierar:, :]

            with torch.no_grad():
                orien_embedding = self.clip_model.token_embedding(oprompt).type(self.dtype)
            orien_ctx_vectors = orien_embedding[0, 1: 1 + n_hierar, :]
            orien_ctx = nn.Parameter(orien_ctx_vectors)
            orien_prefix = span_embedding[:, :1, :]
            orien_suffix = span_embedding[:, 1 + n_hierar:, :]

            span_prompts = self.prompt_learner.construct_prompts(span_ctx.unsqueeze(0), span_prefix, span_suffix)
            span_text_features = self.text_encoder(span_prompts, sprompt, deep_compound_prompts_text)
            span_text_features = self.prompt_learner.proj(span_text_features)
            span_fea_lst.append(span_text_features.unsqueeze(0))

            orien_prompts = self.prompt_learner.construct_prompts(orien_ctx.unsqueeze(0), orien_prefix, orien_suffix)
            orien_text_features = self.text_encoder(orien_prompts, oprompt, deep_compound_prompts_text)
            orien_text_features = self.prompt_learner.proj(orien_text_features)
            orien_fea_lst.append(orien_text_features.unsqueeze(0))
            # ===================== hierarchy =====================
            layout_fea_lst.append(span_text_features.unsqueeze(0))


        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), layout_fea_lst, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tmp_fea = text_features.clone()
        text_features = tmp_fea[:num_classes]
        text_features2 = tmp_fea[num_classes:]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        logits2 = logit_scale * image_features @ text_features2.t()

        span_features = torch.cat(span_fea_lst, dim=0).squeeze()
        span_features = self.prompt_learner.layout_proj(span_features)
        span_features = span_features / span_features.norm(dim=-1, keepdim=True)
        span_logits = logit_scale * span_features @ text_features.t()

        orien_features = torch.cat(orien_fea_lst, dim=0).squeeze()
        orien_features = self.prompt_learner.layout_proj2(orien_features)
        orien_features = orien_features / orien_features.norm(dim=-1, keepdim=True)
        orien_logits = logit_scale * orien_features @ text_features2.t()


        if self.prompt_learner.training:
            targets, targets_top, indices_top = prepare_targets(label)

            return (0.4 * F.cross_entropy(logits, label) + 0.5 * F.cross_entropy(span_logits, label) +
                    0.05 * F.cross_entropy(logits2, targets_top) + 0.05 * F.cross_entropy(orien_logits, targets_top))

        scores = F.softmax(logits * 0.4 + span_logits * 0.5, dim=-1)
        scores2 = F.softmax(logits2 * 0.05 + orien_logits * 0.05, dim=-1)

        scores[:, 0:3] += scores2[:, :1]
        scores[:, 3:5] += scores2[:, 1:2]
        scores[:, 5:6] += scores2[:, 2:3]

        return scores

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class GEOPARSE(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.GEOPARSE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.GEOPARSE.PREC == "fp32" or cfg.TRAINER.GEOPARSE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("GeoParseLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GEOPARSE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.GEOPARSE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, fnames=batch["impath"])
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label, fnames=batch["impath"])
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def model_inference(self, input, fnames=None):
        return self.model(input, fnames=fnames)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input, fnames=batch["impath"])

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
