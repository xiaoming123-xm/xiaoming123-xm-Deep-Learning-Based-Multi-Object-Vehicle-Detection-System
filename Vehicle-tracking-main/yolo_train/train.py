import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import math  # 添加缺失的导入

# ---------------------------- 1. 模型定义（YOLOv8 核心组件）----------------------------

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, k=(3,3), e=1.0) for _ in range(n)])
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
    def forward(self, x):
        # x shape: (b, 4*reg_max, a)
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2,1).softmax(1)).view(b, 4, a)

class Detect(nn.Module):
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            b, _, h, w = x[i].shape
            cls = self.cv3[i](x[i]).view(b, self.nc, -1).permute(0, 2, 1)        # (b, a, nc)
            reg = self.cv2[i](x[i]).view(b, 4 * self.reg_max, -1).permute(0, 2, 1)  # (b, a, 4*reg_max)
            if i == 0:
                cls_out, reg_out = cls, reg
            else:
                cls_out = torch.cat([cls_out, cls], 1)
                reg_out = torch.cat([reg_out, reg], 1)
        # 修正：reg_out 需要转置为 (b, 4*reg_max, a) 再送入 dfl
        bbox = self.dfl(reg_out.permute(0, 2, 1)) * self.stride.view(1, -1, 1).to(reg_out.device)
        bbox = bbox.permute(0, 2, 1)  # 转回 (b, a, 4)
        return torch.cat([bbox, cls_out.sigmoid()], 2)

class YOLOv8(nn.Module):
    def __init__(self, nc=80, ch=3):
        super().__init__()
        # 骨干
        self.conv1 = Conv(ch, 64, 3, 2)
        self.conv2 = Conv(64, 128, 3, 2)
        self.c2f1 = C2f(128, 128, 3, True)
        self.conv3 = Conv(128, 256, 3, 2)
        self.c2f2 = C2f(256, 256, 6, True)
        self.conv4 = Conv(256, 512, 3, 2)
        self.c2f3 = C2f(512, 512, 6, True)
        self.conv5 = Conv(512, 1024, 3, 2)
        self.c2f4 = C2f(1024, 1024, 3, True)
        self.sppf = SPPF(1024, 1024, 5)

        # 颈部
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lat1 = Conv(1024, 512, 1)
        self.c2f5 = C2f(1024, 512, 3, False)
        self.lat2 = Conv(512, 256, 1)
        self.c2f6 = C2f(512, 256, 3, False)
        self.down1 = Conv(256, 256, 3, 2)
        self.c2f7 = C2f(512, 512, 3, False)
        self.down2 = Conv(512, 512, 3, 2)
        self.c2f8 = C2f(1024, 1024, 3, False)

        # 检测头
        ch = [256, 512, 1024]
        self.detect = Detect(nc, ch)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = self.c2f1(x2)
        x3 = self.conv3(x2)
        x3 = self.c2f2(x3)
        x4 = self.conv4(x3)
        x4 = self.c2f3(x4)
        x5 = self.conv5(x4)
        x5 = self.c2f4(x5)
        x5 = self.sppf(x5)

        p5 = self.lat1(x5)
        p4 = self.upsample(p5)
        p4 = torch.cat([p4, x4], 1)
        p4 = self.c2f5(p4)

        p3 = self.upsample(self.lat2(p4))
        p3 = torch.cat([p3, x3], 1)
        p3 = self.c2f6(p3)

        n3 = self.down1(p3)
        n3 = torch.cat([n3, p4], 1)
        n3 = self.c2f7(n3)

        n4 = self.down2(n3)
        n4 = torch.cat([n4, p5], 1)
        n4 = self.c2f8(n4)

        out = self.detect([p3, n3, n4])
        return out

# ---------------------------- 2. 标签分配 ----------------------------

def bbox2dist(anchor_points, bbox, reg_max):
    x1, y1, x2, y2 = bbox.chunk(4, -1)
    return torch.cat([anchor_points[...,0] - x1, y1 - anchor_points[...,1],
                      x2 - anchor_points[...,0], anchor_points[...,1] - y2], -1).clamp(0, reg_max - 0.01)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat([x1y1, x2y2], dim)

def box_iou(box1, box2):
    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])
    inter = (torch.min(box1[:,None,2], box2[:,2]) - torch.max(box1[:,None,0], box2[:,0])).clamp(0) * \
            (torch.min(box1[:,None,3], box2[:,3]) - torch.max(box1[:,None,1], box2[:,1])).clamp(0)
    return inter / (area1[:,None] + area2 - inter + 1e-9)

class TaskAlignedAssigner:
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        b, a, c = pd_scores.shape
        n = gt_bboxes.shape[1]

        pd_bboxes_ = pd_bboxes.unsqueeze(2).repeat(1,1,n,1)
        gt_bboxes_ = gt_bboxes.unsqueeze(1).repeat(1,a,1,1)
        iou = box_iou(pd_bboxes_.view(-1,4), gt_bboxes_.view(-1,4)).view(b,a,n)
        align_metric = pd_scores.unsqueeze(2).pow(self.alpha) * iou.pow(self.beta)
        topk_mask = torch.topk(align_metric, self.topk, dim=1, largest=True)[1]
        assigned = torch.zeros(b, a, dtype=torch.long, device=pd_scores.device) - 1
        for bi in range(b):
            for gi in range(n):
                if mask_gt[bi, gi] == 0: continue
                topk_idx = topk_mask[bi, :, gi]
                best_idx = topk_idx[align_metric[bi, topk_idx, gi].argmax()]
                assigned[bi, best_idx] = gi
        target_labels = torch.zeros((b, a), dtype=torch.long, device=pd_scores.device)
        target_bboxes = torch.zeros((b, a, 4), device=pd_scores.device)
        for bi in range(b):
            assigned_gi = assigned[bi]
            valid = assigned_gi >= 0
            target_labels[bi, valid] = gt_labels[bi, assigned_gi[valid], 0].long()
            target_bboxes[bi, valid] = gt_bboxes[bi, assigned_gi[valid]]
        return target_labels, target_bboxes, assigned

# ---------------------------- 3. 损失函数 ----------------------------

class DFLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
    def forward(self, pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl, reduction='none') * wl +
                F.cross_entropy(pred_dist, tr, reduction='none') * wr).mean()

class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        pred_dist = pred_dist[fg_mask]
        target_bboxes = target_bboxes[fg_mask]
        anchor_points = anchor_points[fg_mask]
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes, xywh=True, CIoU=True)
        loss_iou = (1.0 - iou).mean()
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
        loss_dfl = self.dfl_loss(pred_dist, target_ltrb) / max(target_scores_sum, 1)
        return loss_iou, loss_dfl

def bbox_iou(box1, box2, xywh=True, CIoU=False):
    if xywh:
        b1_x1, b1_x2 = box1[:,0] - box1[:,2]/2, box1[:,0] + box1[:,2]/2
        b1_y1, b1_y2 = box1[:,1] - box1[:,3]/2, box1[:,1] + box1[:,3]/2
        b2_x1, b2_x2 = box2[:,0] - box2[:,2]/2, box2[:,0] + box2[:,2]/2
        b2_y1, b2_y2 = box2[:,1] - box2[:,3]/2, box2[:,1] + box2[:,3]/2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1*h1 + w2*h2 - inter + 1e-9
    iou = inter / union
    if CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2/h2) - torch.atan(w1/h1), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + 1e-7))
        return iou - ( (cw**2+ch**2)/(torch.max(cw,ch)**2) + alpha*v )
    return iou

class ComputeLoss:
    def __init__(self, model, assigner):
        self.model = model
        self.assigner = assigner
        self.reg_max = model.detect.reg_max
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss(self.reg_max)

    def __call__(self, pred, targets):
        b, a = pred.shape[:2]
        # 分离预测
        pred_dist = pred[:, :, :4*self.reg_max].view(b, a, 4, self.reg_max)
        # 注意：dfl 需要输入 (b, 4*reg_max, a)
        pred_bboxes = dist2bbox(self.model.detect.dfl(pred_dist.view(b, a, 4*self.reg_max).permute(0,2,1)),
                                self.anchor_points, xywh=True)
        pred_scores = pred[:, :, 4*self.reg_max:]

        gt_bboxes = targets['bbox']
        gt_labels = targets['cls']
        mask_gt = (gt_bboxes.sum(dim=-1) > 0).float().unsqueeze(-1)

        target_labels, target_bboxes, _ = self.assigner(
            pred_scores.detach(), pred_bboxes.detach(), self.anchor_points,
            gt_labels, gt_bboxes, mask_gt)

        target_scores = torch.zeros((b, a, self.model.detect.nc), device=pred.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        fg_mask = target_scores.sum(dim=-1) > 0

        loss_cls = self.bce(pred_scores, target_scores).sum() / max(fg_mask.sum(), 1)
        loss_iou, loss_dfl = self.bbox_loss(pred_dist.view(b, a, -1), pred_bboxes, self.anchor_points,
                                            target_bboxes, target_scores, fg_mask.sum(), fg_mask)
        return loss_cls + loss_iou + loss_dfl

# ---------------------------- 4. 数据加载 ----------------------------

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        self.label_files = [self.label_dir / (f.stem + '.txt') for f in self.img_files]

    def __len__(self):
        return len(self.img_files)

    def load_image(self, idx):
        img = cv2.imread(str(self.img_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_labels(self, idx):
        labels = []
        if self.label_files[idx].exists():
            with open(self.label_files[idx]) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, w, h = map(float, parts)
                        labels.append([cls, xc, yc, w, h])
        return np.array(labels) if labels else np.zeros((0,5))

    def __getitem__(self, idx):
        img = self.load_image(idx)
        labels = self.load_labels(idx)
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2,0,1) / 255.0
        if len(labels):
            labels[:, [1,3]] *= self.img_size / w0
            labels[:, [2,4]] *= self.img_size / h0
        img = torch.from_numpy(img).float()
        labels = torch.from_numpy(labels) if len(labels) else torch.zeros((0,5))
        return img, labels

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    max_len = max(len(l) for l in labels)
    padded_labels = torch.zeros(len(labels), max_len, 5)
    for i, l in enumerate(labels):
        if len(l):
            padded_labels[i, :len(l)] = l
    return imgs, padded_labels

# ---------------------------- 5. 训练主循环 ----------------------------

def train_yolov8(data_yaml, epochs=300, batch=16, imgsz=640, device='cuda'):
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    # 注意：请根据实际目录结构调整 label 路径
    train_dataset = YOLODataset(data['train'], data['train'].replace('images', 'labels'), imgsz, augment=True)
    train_loader = DataLoader(train_dataset, batch, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = YOLOv8(nc=data['nc']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 生成 anchor_points（所有检测层的网格点）
    anchor_points = []
    strides = [8, 16, 32]
    for stride in strides:
        h = w = imgsz // stride
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        points = torch.stack([x, y], dim=-1).float() * stride
        anchor_points.append(points.view(-1, 2))
    anchor_points = torch.cat(anchor_points, 0).to(device)
    model.detect.stride = torch.tensor(strides, device=device)
    assigner = TaskAlignedAssigner(num_classes=data['nc'])
    loss_fn = ComputeLoss(model, assigner)
    loss_fn.anchor_points = anchor_points

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            gt_bboxes = []
            gt_labels = []
            for b in range(targets.shape[0]):
                valid = targets[b, :, 0] >= 0
                labels = targets[b, valid]
                if len(labels):
                    boxes = torch.zeros((len(labels), 4), device=device)
                    boxes[:, 0] = (labels[:, 1] - labels[:, 3] / 2) * imgsz  # x1
                    boxes[:, 1] = (labels[:, 2] - labels[:, 4] / 2) * imgsz  # y1
                    boxes[:, 2] = (labels[:, 1] + labels[:, 3] / 2) * imgsz  # x2
                    boxes[:, 3] = (labels[:, 2] + labels[:, 4] / 2) * imgsz  # y2
                    gt_bboxes.append(boxes)
                    gt_labels.append(labels[:, 0:1])
                else:
                    gt_bboxes.append(torch.zeros((0, 4), device=device))
                    gt_labels.append(torch.zeros((0, 1), device=device))
            max_gt = max(len(b) for b in gt_bboxes)
            padded_bbox = torch.zeros(len(gt_bboxes), max_gt, 4, device=device)
            padded_label = torch.zeros(len(gt_labels), max_gt, 1, device=device) - 1
            for i, (box, lab) in enumerate(zip(gt_bboxes, gt_labels)):
                padded_bbox[i, :len(box)] = box
                padded_label[i, :len(lab)] = lab
            targets_dict = {'bbox': padded_bbox, 'cls': padded_label}

            pred = model(imgs)
            loss = loss_fn(pred, targets_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}, Avg Loss: {avg_loss:.4f}')
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'yolov8_epoch{epoch}.pt')
    torch.save(model.state_dict(), 'yolov8_final.pt')

if __name__ == '__main__':
    train_yolov8('data.yaml', epochs=300, batch=16, imgsz=640, device='cuda')