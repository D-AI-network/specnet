
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from data import grid_coordinates, grid_neighbors


def robust_z(x, eps=1e-6):
    median = x.median(dim=1, keepdim=True).values
    mad = (x - median).abs().median(dim=1, keepdim=True).values
    return (x - median) / (1.4826 * mad + eps)


def residual_frames(x):
    base = x.clone()
    if x.shape[1] > 1:
        base[:, 1] = x[:, 0]
    if x.shape[1] > 2:
        base[:, 2:] = (x[:, 1:-1] + x[:, :-2] + x[:, 2:]) / 3.0
    return x - base


def temporal_change_gate(x, lags=None, ema=None):
    lags = cfg.LAGS if lags is None else lags
    decay = cfg.EMA_COEF if ema is None else ema
    r = residual_frames(x)
    b, p = r.shape[:2]
    differences = []
    for lag in lags:
        if lag < 1:
            raise ValueError('Each lag must be positive')
        if lag >= p:
            d = r.new_zeros((b, p))
        else:
            # Invalid indices remain zero (not clamped to the first residual).
            d = r.new_zeros((b, p))
            d[:, lag:] = (r[:, lag:] - r[:, :-lag]).abs().mean(dim=(2,3,4))
        differences.append(d)
    score = torch.stack(differences, dim=0).mean(dim=0)
    score = robust_z(score)
    smoothed = [score[:, 0]]
    for t in range(1, p):
        smoothed.append(decay * smoothed[-1] + (1 - decay) * score[:, t])
    return torch.sigmoid(torch.stack(smoothed, dim=1)).detach()


class MultiPeriodicEmbedding(nn.Module):
    def __init__(self, dimension, periods):
        super().__init__()
        if not periods:
            raise ValueError('PERIODS must not be empty')
        self.periods = tuple(periods)
        splits = [dimension // len(periods) + (j < dimension % len(periods)) for j in range(len(periods))]
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(2, split), nn.GELU(), nn.Linear(split, split))
            for split in splits
        ])

    def forward(self, b, p, device):
        position = torch.arange(p, dtype=torch.float32, device=device)[None, :].expand(b,-1)
        parts = []
        for period, layer in zip(self.periods, self.layers):
            phase = 2 * math.pi * position / period
            parts.append(layer(torch.stack((phase.sin(), phase.cos()), dim=-1)))
        return torch.cat(parts, dim=-1)


class SpectralFrequencyEncoder(nn.Module):
    def __init__(self, dim, max_bins, actual_bins):
        super().__init__()
        self.max_bins = max_bins
        self.actual_bins = actual_bins
        self.processors = nn.ModuleList([
            nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(actual_bins)
        ])
        self.combiner = nn.Sequential(nn.Linear(dim * max_bins, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, embedding):
        b, p, n, dim = embedding.shape
        spectrum = torch.fft.rfft(embedding, dim=1)
        used = spectrum.shape[1]
        if used != self.actual_bins:
            raise ValueError(f'Expected {self.actual_bins} rFFT bins, got {used}')
        if used > self.max_bins:
            raise ValueError(f'P={p} produces {used} rFFT bins, exceeding combiner capacity {self.max_bins}')
        mapped = []
        for k in range(used):
            complex_bin = spectrum[:, k]
            stacked = torch.cat((complex_bin.real, complex_bin.imag), dim=-1)
            mapped.append(self.processors[k](stacked))
        if used < self.max_bins:
            mapped.extend([mapped[0].new_zeros((b, n, dim)) for _ in range(self.max_bins - used)])
        return self.combiner(torch.cat(mapped, dim=-1))


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, channels, dim, p):
        super().__init__()
        self.input_project = nn.Linear(channels, dim)
        self.periodic = MultiPeriodicEmbedding(dim//2, cfg.PERIODS)
        self.global_project = nn.Linear(channels, dim//2)
        self.sfe = SpectralFrequencyEncoder(dim, cfg.NUM_FREQUENCIES, p//2+1)
        self.temporal_logits = nn.Parameter(torch.randn(p, 1))

    def forward(self, x, gating_raw=None):
        b,p,h,w,c = x.shape
        node_input = x.reshape(b,p,h*w,c)
        periodic = self.periodic(b,p,x.device)
        global_mean = node_input.mean(dim=2)
        time_embedding = torch.cat((periodic, self.global_project(global_mean)), dim=-1)
        initial = self.input_project(node_input) + time_embedding.unsqueeze(2)
        spectral_feature = self.sfe(initial)  # (B,N,D); one per frequency window
        adjusted = initial + 0.3 * spectral_feature.unsqueeze(1)
        g = temporal_change_gate(x if gating_raw is None else gating_raw)
        base = torch.softmax(self.temporal_logits.squeeze(-1), dim=0)
        weights = torch.softmax(base[None,:] * (1 + cfg.TGATING_GAIN*g), dim=1)
        return (adjusted * weights[:,:,None,None]).sum(dim=1)


class GlobalPrototypeAlignment(nn.Module):
    def __init__(self, h, w, dim, num_prototypes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim)*0.02)
        self.prototype_xy = nn.Parameter(torch.randn(num_prototypes, 2)*0.1)
        self.register_buffer('coordinates', grid_coordinates(h,w,torch.device('cpu')))

    def forward(self, z):
        xy = self.coordinates
        proto_xy = self.prototype_xy.tanh()
        geo_distance = torch.cdist(xy, proto_xy)
        cos_distance = 1 - F.normalize(z,dim=-1) @ F.normalize(self.prototypes,dim=-1).t()
        cost = cfg.GEO_WEIGHT * geo_distance.unsqueeze(0) + cos_distance
        assignment = torch.softmax(-cfg.GPA_TAU * cost/cfg.GPA_EPS,dim=-1)
        refined = z + assignment @ self.prototypes
        return refined, assignment


class MultiScaleGraphAttention(nn.Module):
    def __init__(self, h,w,dim,k):
        super().__init__()
        self.h,self.w,self.dim,self.k=h,w,dim,k
        ids, rel, dist = grid_neighbors(h,w,k,torch.device('cpu'))
        self.register_buffer('neighbor_index', ids)
        self.register_buffer('relative_xy', rel)
        self.register_buffer('normalized_distance', dist)
        self.q = nn.Linear(dim,dim)
        self.k_proj = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        # Paper: the constant 1/K assignment first moment is omitted.
        self.concentration_net = nn.Sequential(nn.Linear(1,16),nn.GELU(),nn.Linear(16,16),nn.GELU())
        self.node_net = nn.Sequential(nn.Linear(dim,16),nn.GELU())
        self.anisotropic_head = nn.Linear(32,3)
        self.scale_head = nn.Linear(32,3)

    def forward(self,h,assignments):
        b,n,d=h.shape
        idx=self.neighbor_index
        rel=self.relative_xy
        dist=self.normalized_distance
        if len(idx)!=n:
            raise ValueError('Spatial grid shape does not match model construction')
        q=self.q(h)
        key=self.k_proj(h)[:,idx,:]
        value=self.v(h)[:,idx,:]
        neighboring_assignments=assignments[:,idx,:]
        concentration=assignments.square().mean(dim=-1,keepdim=True)
        condition=torch.cat([self.concentration_net(concentration),self.node_net(h)],dim=-1)
        raw_a,raw_b,raw_c=self.anisotropic_head(condition).unbind(-1)
        # Lower triangular L; diag(exp(a), exp(c)) -> A=L@L^T SPD
        la=raw_a.clamp(-10,10).exp()
        lc=raw_c.clamp(-10,10).exp()
        lb=raw_b
        l=torch.zeros(b,n,2,2,dtype=h.dtype,device=h.device)
        l[...,0,0]=la
        l[...,1,0]=lb
        l[...,1,1]=lc
        metric=l@l.transpose(-1,-2)
        anisotropy=torch.einsum('nki,bnij,nkj->bnk',rel,metric,rel)
        content=(q.unsqueeze(2)*key).sum(dim=-1)/math.sqrt(d)
        structural=(assignments.unsqueeze(2)*neighboring_assignments).sum(dim=-1)
        fused=[]
        for radius,omega0,omega1,center_coeff in zip(cfg.SCALES,cfg.OMEGA0,cfg.OMEGA1,cfg.CENTER_COEF):
            sigma=(omega0+omega1*dist).clamp_min(1e-6)
            center_bias=center_coeff*torch.exp(-dist.square()/cfg.CENTER_WIDTH**2)
            logits=(content+structural-anisotropy/sigma[None,:,:].square()
                    -cfg.ISO_DECAY*dist[None,:,:]+center_bias[None,:,:])
            # Paper Eq. fused logits: 1e4 penalty applied before the scale-gate sum.
            logits=logits-1e4*(dist[None,:,:]>radius).to(logits.dtype)
            fused.append(logits)
        gate=torch.softmax(self.scale_head(condition),dim=-1)
        combined=sum(gate[...,j:j+1]*v for j,v in enumerate(fused))
        attention=torch.softmax(combined,dim=-1)
        output=h + (attention.unsqueeze(-1)*value).sum(dim=2)
        magnitude=metric.norm(dim=(-1,-2)).reshape(b,self.h,self.w)
        variation=(magnitude[:,1:,:]-magnitude[:,:-1,:]).abs().mean() + (magnitude[:,:,1:]-magnitude[:,:,:-1]).abs().mean()
        return output,variation


class SPECNet(nn.Module):
    def __init__(self,h=100,w=100,channels=3,p=25):
        super().__init__()
        if channels<1 or p<2:
            raise ValueError('channels must be >=1 and input length >=2')
        self.h,self.w,self.channels=h,w,channels
        self.stfe=TemporalFeatureExtractor(channels,cfg.D_MODEL,p)
        self.gpa=GlobalPrototypeAlignment(h,w,cfg.D_MODEL,cfg.N_PROTO)
        self.mgat=MultiScaleGraphAttention(h,w,cfg.D_MODEL,cfg.K_NN)
        self.output=nn.Linear(cfg.D_MODEL,channels)

    def forward(self,x,raw_input=None):
        if x.ndim!=5 or x.shape[2:]!=(self.h,self.w,self.channels):
            raise ValueError(f'Expected (B,P,{self.h},{self.w},{self.channels}); got {tuple(x.shape)}')
        z=self.stfe(x,gating_raw=raw_input)
        h,alignment=self.gpa(z)
        result,tv=self.mgat(h,alignment)
        # Equation (one-step): direct node-wise projection, without an extra last-value skip.
        prediction=self.output(result).reshape(x.shape[0],self.h,self.w,self.channels)
        return prediction,tv
