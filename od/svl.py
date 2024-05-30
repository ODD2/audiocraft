import torch
import logging
import vclip as CLIP
import torch.nn as nn
from functools import reduce
from operator import or_
from enum import IntFlag, auto


class OpMode(IntFlag):
    S = auto()  # spatial
    T = auto()  # temporal


def load_model(architecture, **kargs):
    # architecture parameter split
    # e.g: ViT-B/16|laion2b_s34b_b79k
    params = architecture.split('|')

    clip_arch = params[0]
    model, transform = CLIP.load(
        clip_arch,
        "cpu",
        **kargs
    )

    return model, transform


def call_module(module):
    def fn(*args, **kwargs):
        return module(*args, **kwargs)
    return fn


def get_module(module):
    def fn():
        return module
    return fn


class VideoAttrExtractor(nn.Module):
    def __init__(
        self,
        architecture,
        text_embed,
        store_attrs=[],
        attn_record=False,
        pretrain=None
    ):
        super().__init__()
        self.model, self.transform = load_model(
            architecture,
            store_attrs=store_attrs,
            attn_record=attn_record
        )
        self.model = self.model.visual.float()

        if (pretrain):
            logging.info("Loading image encoder pretrain weights...")
            state_dict = torch.load(pretrain, "cpu")
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except:
                conflicts = self.model.load_state_dict(state_dict, strict=False)
                logging.warning(
                    f"during visual pretrain weights loading, disabling strict mode with conflicts:\n{conflicts}"
                )

        self.model.requires_grad_(False)

        if not text_embed:
            self.model.proj = None
            self.feat_dim = self.model.transformer.width
        else:
            self.feat_dim = self.model.output_dim

    @property
    def n_px(self):
        return self.model.input_resolution

    @property
    def n_layers(self):
        return self.model.transformer.layers

    @property
    def n_heads(self):
        return self.model.transformer.heads

    @property
    def patch_size(self):
        return self.model.patch_size

    @property
    def patch_num(self):
        return self.model.patch_num

    @property
    def n_patch(self):
        return int(self.n_px // self.patch_size)

    @property
    def embed_dim(self):
        return self.feat_dim

    def forward(self, x):
        b, t = x.shape[:2]

        # pass throught for attributes
        embeds = self.model(x)
        # retrieve all layer attributes
        layer_attrs = []
        for blk in self.model.transformer.resblocks:
            attrs = blk.pop_attr()
            layer_attrs.append(attrs)
        return dict(
            layer_attrs=layer_attrs,
            embeds=embeds
        )

    def train(self, mode=True):
        self.model.eval()
        return self


class SynoBlock(nn.Module):
    def __init__(
        self,
        n_synos,
        d_model,
        n_head,
        n_patch,
        n_frames,
        ksize_t,
        ksize_s,
        t_attrs,
        s_k_attr,
        s_v_attr,
        op_mode,
        store_attrs=[],
        attn_record=False
    ):
        super().__init__()

        # parameters
        self.n_patch = n_patch
        self.t_attrs = t_attrs
        self.s_k_attr = s_k_attr
        self.s_v_attr = s_v_attr

        self.op_mode = op_mode

        if (OpMode.T in op_mode):
            # modules
            self.t_conv = self.make_2dconv(
                ksize_t,
                sum([
                    1 if attr in ["out", "emb"] else n_head
                    for attr in t_attrs
                ]),
                1
            )

            self.t_proj = nn.Sequential(
                nn.LayerNorm(n_frames**2),
                nn.Linear(
                    n_frames**2,
                    n_frames
                ),
                nn.GELU(),
                nn.Linear(
                    n_frames,
                    n_frames**2
                )
            )

            self.p_conv = self.make_2dconv(
                ksize_s,
                n_frames ** 2,
                1
            )

        if (OpMode.S in op_mode):

            self.syno_embedding = nn.Parameter(
                torch.zeros(n_synos, d_model)
            )

        # attribute storage
        self.store_attrs = store_attrs
        self.attr = {}

        # attention map recording
        self.attn_record = attn_record
        self.aff = None

    def make_2dconv(self, ksize, in_c, out_c, groups=1):
        conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=ksize,
            stride=1,
            padding=ksize // 2,
            groups=groups,
            bias=True
        )

        nn.init.normal_(conv.weight, std=0.001)
        nn.init.zeros_(conv.bias)

        return conv

    def pop_attr(self):
        ret = self.get_attr()
        self.attr.clear()
        return ret

    def get_attr(self):
        return {k: self.attr[k] for k in self.attr}

    def set_attr(self, **attr):
        self.attr = {
            k: attr[k]
            for k in attr
            if k in self.store_attrs
        }

    def temporal_detection(self, attrs):
        b, t, l, h, d = attrs['q'][:, :, 1:].shape  # ignore cls token
        p = self.n_patch  # p = l ** 0.5

        affs = []
        for attr in self.t_attrs:
            _attr = attrs[attr][:, :, 1:]  # ignore cls token

            if (len(_attr.shape) == 4):
                _attr = _attr.unsqueeze(-2)

            _attr = _attr.permute(0, 2, 1, 3, 4)

            aff = torch.einsum(
                'nlqhc,nlkhc->nlqkh',
                _attr / (_attr.size(-1) ** 0.5),
                _attr
            )

            aff = aff.softmax(dim=-2)

            aff = aff.flatten(0, 1)  # shape = (n*l,t,t,h)
            aff = aff.permute(0, 3, 1, 2)  # shape = (n*l,h,t,t)
            affs.append(aff)

        aff = torch.cat(affs, dim=1)  # shape = (n*l, 3*h, t, t)

        aff = self.t_conv(aff)  # shape = (n*l, r, t, t) where r is number of filters

        aff = aff.unflatten(0, (b, p, p)).flatten(3)  # shape = (n, p, p, t*t)
        aff = aff + self.t_proj(aff)
        aff = aff.permute(0, 3, 1, 2)  # shape = (n, t*t, p, p)

        aff = self.p_conv(aff)  # shape = (n, 1, p, p)

        y = aff.flatten(1)

        return dict(y=y)  # shape = (n, p*p)

    def spatial_detection(self, attrs):
        b, t, l, h, d = attrs['q'][:, :, 1:].shape  # ignore cls token

        _k = attrs[self.s_k_attr][:, :, 1:]  # ignore cls token
        _v = attrs[self.s_v_attr][:, :, 1:]  # ignore cls token

        # prepare query
        s_q = self.syno_embedding.unsqueeze(0).repeat(b, 1, 1)  # shape = (b, synos, width)

        if (len(_k.shape) == 5):
            _k = _k.flatten(-2).contiguous()  # match shape

        if (len(_v.shape) == 5):
            _v = _v.flatten(-2).contiguous()  # match shape

        s_aff = torch.einsum(
            'nqw,ntkw->ntqk',
            s_q / (s_q.size(-1) ** 0.5),
            _k
        )

        s_aff = s_aff.softmax(dim=-1)

        s_mix = torch.einsum('ntql,ntlw->ntqw', s_aff, _v)

        y = s_mix.flatten(1, 2).mean(dim=1)  # shape = (b,w)

        if self.attn_record:
            self.aff = s_aff

        return dict(s_q=s_q, y=y)

    def forward(self, attrs):
        self.pop_attr()
        y_t = 0
        y_s = 0
        if OpMode.T in self.op_mode:
            ret_t = self.temporal_detection(attrs)
            y_t = ret_t.pop('y')
            self.set_attr(
                **ret_t
            )
        if OpMode.S in self.op_mode:
            ret_s = self.spatial_detection(attrs)
            y_s = ret_s.pop('y')
            self.set_attr(
                **ret_s
            )

        return y_t, y_s


class SynoDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        num_synos,
        num_frames,
        ksize_s,
        ksize_t,
        t_attrs,
        s_k_attr,
        s_v_attr,
        op_mode,
        store_attrs=[],
        attn_record=False
    ):
        super().__init__()
        d_model = encoder.transformer.width
        n_head = encoder.transformer.heads
        n_patch = int((encoder.patch_num)**0.5)

        self.encoder = get_module(encoder)

        self.decoder_layers = nn.ModuleList([
            SynoBlock(
                n_synos=num_synos,
                d_model=d_model,
                n_head=n_head,
                n_patch=n_patch,
                n_frames=num_frames,
                ksize_t=ksize_t,
                ksize_s=ksize_s,
                t_attrs=t_attrs,
                s_k_attr=s_k_attr,
                s_v_attr=s_v_attr,
                op_mode=op_mode,
                store_attrs=store_attrs,
                attn_record=attn_record
            )
            for _ in range(encoder.transformer.layers)
        ])
        self.op_mode = op_mode

        self.d_model = d_model

        self.out_s_proj = nn.Sequential(
            nn.LayerNorm(d_model)
        )
        self.out_t_proj = nn.Sequential(
            nn.Linear(n_patch**2, d_model, bias=True),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        b = x.shape[0]
        layer_output = dict(
            y_t=[],
            y_s=[]
        )

        # first, we prepare the encoder before the transformer layers.
        x = self.encoder()._prepare(x)

        # now, we alternate between synoptic and encoder layers
        for enc_blk, dec_blk in zip(
            self.encoder().transformer.resblocks,
            self.decoder_layers
        ):
            data = enc_blk(x)
            x = data["emb"]
            y_t, y_s = dec_blk(data)
            layer_output["y_t"].append(y_t)
            layer_output["y_s"].append(y_s)

        # last, we are done with the encoder, therefore skipping the _finalize step.
        # x =  self.encoder()._finalize(x)

        # aggregate the layer outputs
        y_s = sum(layer_output["y_s"])
        y_t = sum(layer_output["y_t"])

        out = []
        if (OpMode.T in self.op_mode):
            y_t = self.out_t_proj(y_t)
            out.append(y_t)

        if (OpMode.S in self.op_mode):
            y_s = self.out_s_proj(y_s)
            out.append(y_s)

        return torch.stack(out, dim=1)


class SynoVideoAttrExtractor(VideoAttrExtractor):
    def __init__(
        self,
        # VideoAttrExtractor
        architecture,
        text_embed,
        pretrain=None,
        store_attrs=[],
        attn_record=False,
        # synoptic
        ksize_t=3,
        ksize_s=3,
        num_synos=1,
        num_frames=1,
        s_k_attr="k",
        s_v_attr="v",
        op_mode=(OpMode.S | OpMode.T),
        t_attrs=["q", "k", "v"],
    ):

        if (type(op_mode) == list):
            op_mode = reduce(or_, [OpMode[i] for i in op_mode])

        assert len(store_attrs) == 0, "stored attrs are not supported in the current model usecase."
        super(SynoVideoAttrExtractor, self).__init__(
            architecture=architecture,
            text_embed=text_embed,
            store_attrs=store_attrs,
            attn_record=attn_record,
            pretrain=pretrain
        )
        self.decoder = SynoDecoder(
            encoder=self.model,
            num_synos=num_synos,
            num_frames=num_frames,
            t_attrs=t_attrs,
            ksize_t=ksize_t,
            ksize_s=ksize_s,
            s_k_attr=s_k_attr,
            s_v_attr=s_v_attr,
            op_mode=op_mode,
            store_attrs=store_attrs,
            attn_record=attn_record
        )

    def forward(self, x):
        raise NotImplementedError()

    def forward(self, x):
        embeds = self.decoder(x=x)

        # layer_attrs = []
        # for enc_blk, dec_blk in zip(self.model.transformer.resblocks, self.decoder.decoder_layers):
        #     layer_attrs.append(
        #         {
        #             **enc_blk.pop_attr(),
        #             **dec_blk.pop_attr()
        #         }
        #     )

        return embeds

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
            self.decoder.train()
        return self
