import torch
import datetime
import numpy as np


def training_loop(n_epochs, optimizer, model, device, loss_fn, train_data, val_data):

    for epoch in range(1, n_epochs + 1):
        loss_train = 0
        skip_counter = 0
        # np.random.shuffle(train_data)
        for img, labels, bbox in train_data:
            if len(labels) > 0:
                image = torch.from_numpy(img).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)

                ret = model(image, bbox, labels)
                if ret is None:
                    skip_counter += 1
                    print(f'Skip {skip_counter} images')
                    continue
                anchor_locations, anchor_labels, pred_anchor_locs, pred_cls_scores, roi_cls_loc, roi_cls_score, gt_roi_locs, gt_roi_labels = ret
                loss = loss_fn(anchor_locations, anchor_labels, pred_anchor_locs, pred_cls_scores, roi_cls_loc, roi_cls_score, gt_roi_locs, gt_roi_labels, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train += loss.item()
        print('{} Epoch {}, total training loss {}'.format(datetime.datetime.now(), epoch, loss_train))