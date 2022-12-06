import torch


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.L1Loss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        #print("targets的类型是：\n",type(targets))
        heatmaps  = torch.stack([t.to(device) for t in targets["heatmap"]])
        # [num_kps] -> [B, num_kps]
        
        #kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss = torch.sum(loss * 1) / bs
        return loss
