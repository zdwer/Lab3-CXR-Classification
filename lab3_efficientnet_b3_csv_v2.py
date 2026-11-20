
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

CLASSES=['normal','bacteria','virus','COVID-19']
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def create_transform(train):
    operations=[transforms.Resize((300,300))]
    if train:
        operations += [transforms.RandomAffine(degrees=5, translate=(0.03,0.03)),
                       transforms.ColorJitter(brightness=0.08, contrast=0.08)]
    operations += [transforms.ToTensor(),
                   transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
    return transforms.Compose(operations)

class CSVDataset(Dataset):

    FILENAME_COLUMNS = ('new_filename', 'filename', 'image_filename', 'image_name',
                        'img_name', 'image', 'file_name', 'file', 'path', 'image_path')
    LABEL_COLUMNS = ('label', 'class', 'target', 'category', 'diagnosis', 'class_name')

    def __init__(self, root, csv_file, train=False):
        self.root = Path(root)
        self.csv_path = Path(csv_file)
        if not self.root.is_dir():
            raise FileNotFoundError(f'Image directory not found: {self.root}')
        if not self.csv_path.is_file():
            raise FileNotFoundError(f'Label CSV not found: {self.csv_path}')
        self.records = []
        with self.csv_path.open('r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            filename_col = next((c for c in self.FILENAME_COLUMNS if c in headers), None)
            if filename_col is None:
                raise ValueError(f'Cannot identify filename column in {self.csv_path}; columns={headers}')
            onehot = all(c in headers for c in CLASSES)
            label_col = next((c for c in self.LABEL_COLUMNS if c in headers), None)
            if not onehot and label_col is None:
                raise ValueError(f'Cannot identify class labels in {self.csv_path}; columns={headers}. '
                                 f'Expected {CLASSES} or one of {self.LABEL_COLUMNS}')
            for line_number, row in enumerate(reader, start=2):
                name = (row.get(filename_col) or '').strip()
                if not name:
                    raise ValueError(f'Empty filename at {self.csv_path}:{line_number}')
                # Dataset is flat, but allow a matching nested relative path.
                candidate = self.root / name
                if not candidate.is_file():
                    candidate = self.root / Path(name).name
                if not candidate.is_file():
                    raise FileNotFoundError(f'Missing image for {self.csv_path}:{line_number}: {name} in {self.root}')
                if onehot:
                    try:
                        flags = [float(row[c]) for c in CLASSES]
                    except (TypeError, ValueError):
                        raise ValueError(f'Non-numeric one-hot class at {self.csv_path}:{line_number}: {row}')
                    if sum(v == 1 for v in flags) != 1 or any(v not in (0, 1) for v in flags):
                        raise ValueError(f'Expected exactly one class=1 at {self.csv_path}:{line_number}: {flags}')
                    label = flags.index(1)
                else:
                    raw = (row.get(label_col) or '').strip()
                    norm = raw.casefold().replace(' ', '').replace('_','').replace('-','')
                    synonyms = {'normal':0, 'healthy':0, 'bacteria':1, 'bacterial':1,
                                'bacterialpneumonia':1, 'virus':2, 'viral':2,
                                'viralpneumonia':2, 'covid19':3, 'covid':3,
                                '0':0, '1':1, '2':2, '3':3}
                    if norm not in synonyms:
                        raise ValueError(f'Unknown label at {self.csv_path}:{line_number}: {raw!r}; '
                                         'please inspect CSV label encoding before training')
                    label = synonyms[norm]
                self.records.append((candidate, label))
        if not self.records:
            raise ValueError(f'No image rows in CSV: {self.csv_path}')
        self.transform = create_transform(train)
        print(f'Loaded {len(self.records)} samples from {self.csv_path}; images={self.root}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, label

def build_model(pretrained):
    weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    net=models.efficientnet_b3(weights=weights)
    channels=net.classifier[-1].in_features
    net.classifier[-1]=nn.Linear(channels,len(CLASSES))
    return net

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.,alpha=None):
        super().__init__(); self.gamma=gamma
        if alpha is not None: self.register_buffer('alpha',alpha)
        else: self.alpha=None
    def forward(self,logits,targets):
        ce=nn.functional.cross_entropy(logits, targets, reduction='none')
        loss=(1-torch.exp(-ce))**self.gamma * ce
        if self.alpha is not None: loss=loss*self.alpha[targets]
        return loss.mean()

def macro_metrics(labels,predictions):
    return float(f1_score(labels,predictions,labels=list(range(4)),average='macro',zero_division=0))

@torch.no_grad()
def validate(net,loader,device,criterion):
    net.eval()
    labels=[]; preds=[]; total=0.; n=0
    for images,y in loader:
        images,y=images.to(device),y.to(device)
        logits=net(images)
        total+=criterion(logits,y).item()*len(y); n+=len(y)
        labels.extend(y.cpu().tolist()); preds.extend(logits.argmax(dim=1).cpu().tolist())
    return total/max(n,1), macro_metrics(labels,preds),labels,preds

def train(args):
    set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set=CSVDataset(args.train_dir,args.train_csv,train=True)
    val_set=CSVDataset(args.val_dir,args.val_csv,train=False)
    counts=Counter(y for _,y in train_set.records)
    val_counts=Counter(y for _,y in val_set.records)
    print('Train class counts:', {CLASSES[i]:counts[i] for i in range(4)})
    print('Validation class counts:', {CLASSES[i]:val_counts[i] for i in range(4)})
    if args.check_data_only:
        print('CSV labels and image paths verified. No model training started.')
        return
    inv_sqrt=torch.tensor([1/np.sqrt(counts[i]) for i in range(4)],dtype=torch.float32,device=device)
    weights=inv_sqrt / inv_sqrt.mean()
    if args.loss=='weighted_ce': criterion=nn.CrossEntropyLoss(weight=weights)
    elif args.loss=='focal': criterion=FocalLoss(gamma=2.0,alpha=weights)
    else: criterion=nn.CrossEntropyLoss()
    net=build_model(not args.no_pretrained).to(device)
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,
                            num_workers=args.num_workers,pin_memory=device.type=='cuda')
    val_loader=DataLoader(val_set,batch_size=args.batch_size,shuffle=False,
                          num_workers=args.num_workers,pin_memory=device.type=='cuda')
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    history=[]; best=-1.; stale=0
    print(f'Device={device}; train={len(train_set)}; val={len(val_set)}; class counts={dict(counts)}')
    for epoch in range(1,args.epochs+1):
        net.train(); running=0.; n=0
        for images,y in train_loader:
            images,y=images.to(device,non_blocking=True),y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=device.type=='cuda'):
                logits=net(images); loss=criterion(logits,y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running+=loss.item()*len(y); n+=len(y)
        val_loss,macro_f1,y_true,y_pred=validate(net,val_loader,device,criterion)
        scheduler.step()
        row={'epoch':epoch,'train_loss':round(running/n,6),'val_loss':round(val_loss,6),
             'val_macro_f1':round(macro_f1,6)}
        history.append(row)
        with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerows(history)
        print(row)
        if macro_f1>best+1e-6:
            best=macro_f1;stale=0
            torch.save({'model_state_dict':net.state_dict(),'classes':CLASSES,
                        'epoch':epoch,'val_macro_f1':best,'loss':args.loss},out/'best_model.pt')
            p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,labels=list(range(4)),zero_division=0)
            details={'epoch':epoch,'val_macro_f1':best,
                     'per_class':{cls:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f1[i])}
                                  for i,cls in enumerate(CLASSES)},
                     'confusion_matrix':confusion_matrix(y_true,y_pred,labels=list(range(4))).tolist()}
            (out/'best_val_metrics.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        else: stale+=1
        if stale>=args.patience:
            print('Early stopping triggered'); break
    print(f'Best validation Macro-F1: {best:.5f}; checkpoint: {out/"best_model.pt"}')

@torch.no_grad()
def predict(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if list(checkpoint.get('classes',[]))!=CLASSES:
        raise ValueError('Checkpoint classes do not match Kaggle submission class order')
    model=build_model(False).to(device)
    model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    with Path(args.sample_csv).open('r',newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['new_filename']+CLASSES:
            raise ValueError(f'Invalid sample CSV header: {reader.fieldnames}')
        filenames=[row['new_filename'] for row in reader]
    if not filenames or len(set(filenames))!=len(filenames):
        raise ValueError('Sample CSV filenames must be nonempty and unique')
    testdir=Path(args.test_dir)
    missing=[name for name in filenames if not (testdir/name).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing test images; first few: {missing[:5]}')
    transform=create_transform(False)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for offset in range(0,len(filenames),args.batch_size):
        group=filenames[offset:offset+args.batch_size]
        images=[]
        for name in group:
            with Image.open(testdir/name) as im:
                images.append(transform(im.convert('RGB')))
        logits=model(torch.stack(images).to(device))
        labels=logits.argmax(dim=1).cpu().tolist()
        for name,pred in zip(group,labels):
            rows.append({'new_filename':name,**{cls:int(i==pred) for i,cls in enumerate(CLASSES)}})
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['new_filename']+CLASSES)
        writer.writeheader();writer.writerows(rows)
    assert [r['new_filename'] for r in rows]==filenames
    assert all(sum(r[c] for c in CLASSES)==1 for r in rows)
    print(f'Wrote {len(rows)} model-predicted rows: {output}')

def main():
    parser=argparse.ArgumentParser(description='CXR four-class EfficientNet-B3 for Kaggle Macro-F1')
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    t.add_argument('--train-dir',required=True);t.add_argument('--train-csv',required=True)
    t.add_argument('--val-dir',required=True);t.add_argument('--val-csv',required=True)
    t.add_argument('--output-dir',default='runs/lab3');t.add_argument('--loss',choices=['ce','weighted_ce','focal'],default='weighted_ce')
    t.add_argument('--epochs',type=int,default=25);t.add_argument('--batch-size',type=int,default=16)
    t.add_argument('--num-workers',type=int,default=4);t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--check-data-only',action='store_true',help='Validate image paths and CSV labels without training')
    t.add_argument('--patience',type=int,default=7);t.add_argument('--seed',type=int,default=42)
    t.add_argument('--no-pretrained',action='store_true',help='Avoid downloading pretrained weights (usually worse)')
    p=sub.add_parser('predict')
    p.add_argument('--test-dir',required=True);p.add_argument('--sample-csv',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',default='submission.csv')
    p.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.command=='train':train(args)
    else:predict(args)

if __name__=='__main__':main()
