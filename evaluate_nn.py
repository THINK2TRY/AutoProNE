import numpy as np
import torch
import torch.nn as nn

import utils
from utils import LogReg

hid_units = 512

xent = nn.CrossEntropyLoss()
cnt_wait = 0


def evaluate_pre(args, emb):
    _, _, labels, idx_train, idx_val, idx_test = utils.load_data(args.dataset)

    nb_classes = int(labels.shape[1])

    labels = torch.FloatTensor(labels[np.newaxis])

    emb = torch.FloatTensor(emb)
    if torch.cuda.is_available():
        emb = emb.cuda()
        labels = labels.cuda()

    train_embs = emb[idx_train]
    test_embs = emb[idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    evaluate_semi(train_embs=train_embs,
                  test_embs=test_embs,
                  train_lbls=train_lbls,
                  test_lbls=test_lbls,
                  nb_classes=nb_classes)


def evaluate_semi(train_embs, test_embs, train_lbls, test_lbls, nb_classes):
    is_cuda = torch.cuda.is_available()
    # torch.manual_seed(0)
    # if is_cuda:
    #     torch.cuda.manual_seed(0)
    hid_units = train_embs.shape[1]
    tot = torch.zeros(1)
    if is_cuda:
        tot = tot.cuda()
    accs = []
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        if is_cuda:
            log.cuda()
        pat_steps = 0
        best_acc = torch.zeros(1)
        if is_cuda:
            best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print(acc)
        tot += acc
    print('Average accuracy:', tot / 50)
    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())
