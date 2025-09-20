import torch
import torch.nn.functional as F

class SegGradCAM:
    def __init__(self, model):
        self.model = model
        self.fmaps = None
        self.grads = None
        self.h = model.enc[-1].register_forward_hook(self._fh)
        self.g = model.enc[-1].register_full_backward_hook(self._bh)

    def _fh(self, m, i, o):
        self.fmaps = o.detach()

    def _bh(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x):
        self.model.zero_grad()
        logits = self.model(x)
        target = torch.sigmoid(logits).mean()
        target.backward()
        weights = self.grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.fmaps).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min())/(cam.max()-cam.min()+1e-6)
        return cam.squeeze(0).squeeze(0).cpu().numpy()
