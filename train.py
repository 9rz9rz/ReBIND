import argparse
import numpy as np
import pandas as pd
import random
import torch
import os
import time
import logging
import json

import torch.utils
import torch.utils.data
from torch.optim.lr_scheduler import OneCycleLR
from lavis.models import load_model_and_preprocess
from utility import base_path, device, params
import utility
from tqdm import tqdm
import copy
import data
from lavis.models.blip2_models.cirmodel import CIRModel
from precompute_evaluation import evaluate_features

def set_seed(seed: int = 42, shuffle_seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(shuffle_seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def blip_finetune(args):
    blip_model_name = 'blip2_cir_image_diff_features'
    if args.exp_name:
        training_path = base_path / 'log' / f'{args.dataset}'/ args.exp_name
    else:    
        training_path = base_path / 'log' / f'{args.dataset}'/ args.timestamp
    os.makedirs(training_path, exist_ok=True)
    backbone = args.backbone
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    blip_model.inn_rho = args.inn_rho
    blip_model.inn_sinkhorn_reg = args.inn_sinkhorn_reg
    blip_model.inn_sinkhorn_iter = args.inn_sinkhorn_iter
    blip_model.inn_ortho_coef = args.inn_ortho_coef
    blip_model.swap_warmup_epochs = args.swap_warmup_epochs
    blip_model.lrm = args.lrm
    blip_model.inference_balance = args.inference_balance
    blip_model.similarity_option = args.similarity_option
    blip_model.aux_loss_option = args.aux_loss_option

    blip_model.build_inn(args.inn_block_dim)

    blip_model.log_cfg()
    
    preprocess = utility.targetpad_transform(target_ratio=1.25, dim=224)
    
    dataset = data.get_dataset(args.dataset, preprocess, 'train', mode='relative', noise_ratio=args.noise_ratio, args=args)
    learning_rate = args.lr
    num_epochs = args.num_epochs
    loss_balance_dict = {
        'lpm': args.lpm,
        'lsa': args.lsa,
        'lrd': args.lrd,
        'lmask': args.swap_coef_bce,
        'lswap_main': args.swap_coef_main,
        'lswap_aux1': args.swap_coef_aux1,
        'lswap_aux2': args.swap_coef_aux2,
        'lortho': args.inn_ortho_coef,
    }
    
    optimizer = torch.optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':args.weight_decay}])
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, 
                           div_factor=100., steps_per_epoch=len(dataloader), epochs=num_epochs)
    
    scaler = torch.cuda.amp.GradScaler()

    training_log_frame = pd.DataFrame()
    accuracy_list = []
    best_acc = 0
    dataset.load_image_features()
    
    valset = data.get_dataset(args.dataset, preprocess, 'val', mode='relative', args=args)
    valset.load_image_features()
    
    for epoch in range(num_epochs):
        warmup = None
        pn_loss = copy.deepcopy(args.pn_loss)

        
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        partitioner_type = args.partitioner
        if warmup:
            partitioner_type = 'all_positive'
            pn_loss['positive_loss'] = pn_loss['warmup_loss']
            pn_loss['negative_loss'] = 'None'
            pn_loss['positive_align_loss'] = pn_loss['warmup_align_loss']
            pn_loss['negative_align_loss'] = 'None'
            pn_loss['trade_off'] = 1.0
            pn_loss['trade_off_align'] = 1.0
                   
        partitioner = utility.Partitioner(partitioner_type, args.split_type, args.threshold,
                                          timestamp=args.timestamp, epoch=epoch, dataset_name=args.dataset)
        label_mask = partitioner.fit_features(blip_model, dataloader, txt_processors)
        label_mask = label_mask.to(device)

        warmup_end = args.swap_warmup_epochs
        if epoch < warmup_end:
            blip_model.swap_alpha = 0.0
        else:
            swap_epoch = epoch - warmup_end
            blip_model.swap_alpha = min(1.0, swap_epoch / max(1, args.swap_warmup_epochs))

        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(dataloader, ncols=120, mininterval=30)
        train_bar.set_description(desc=f"[{epoch+1}/{num_epochs}]")
        for reference_name, target_hard_name, captions, index in train_bar:
            reference_images = dataset.get_image_features(reference_name).to(device, non_blocking=True)
            target_images = dataset.get_image_features(target_hard_name).to(device, non_blocking=True)
            optimizer.zero_grad()
            labels = label_mask[index]
            if args.dataset == 'FashionIQ':
                flattened_captions = np.array(captions).T.flatten().tolist()
                captions = utility.generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors['eval'](caption) for caption in captions]
            blip_model.train()
            samples = {"image": reference_images, "target": target_images, "text_input":captions}
            with torch.cuda.amp.autocast():
                loss_dict = blip_model(samples, labels, pn_loss, warmup)
            loss = 0.
            for key in loss_dict:
                if key in loss_balance_dict:
                    loss += loss_balance_dict[key] * loss_dict[key]
                else:
                    raise ValueError('loss type is invalid')

            if not torch.isfinite(loss):
                logging.warning(f"Skipping batch: loss is {loss.item():.3f}, skipping update")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(blip_model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            images_in_batch = reference_images.shape[0]
            for key in loss_dict.keys():
                if key not in train_running_results:
                    train_running_results[key] = 0
                train_running_results[key] += loss_dict[key].to('cpu').detach().item() * images_in_batch

            balanced_batch = {}
            for key in loss_dict:
                if key in loss_balance_dict:
                    balanced_batch[key] = loss_balance_dict[key] * loss_dict[key].to('cpu').detach().item() * images_in_batch
            balanced_batch['total_loss'] = sum(balanced_batch.values())
            for key, val in balanced_batch.items():
                bl_key = f'balanced_{key}'
                if bl_key not in train_running_results:
                    train_running_results[bl_key] = 0
                train_running_results[bl_key] += val

            train_running_results['images_in_epoch'] += images_in_batch


        parts = [f"{k}: {v / train_running_results['images_in_epoch']:.3f}"
                 for k, v in train_running_results.items() if k != 'images_in_epoch']
        tqdm.write(f"[{epoch+1}/{num_epochs} done] " + ", ".join(parts))
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv((training_path / 'train_metrics.csv'), index=False)

        balanced_log_dict = {'epoch': epoch, 'total_loss': float(
            train_running_results.get('balanced_total_loss', 0) / train_running_results['images_in_epoch']
        )}
        for key in loss_balance_dict:
            bl_key = f'balanced_{key}'
            if bl_key in train_running_results:
                balanced_log_dict[key] = float(train_running_results[bl_key] / train_running_results['images_in_epoch'])
        balanced_log_frame = pd.DataFrame(data=balanced_log_dict, index=[0])
        balanced_csv_path = training_path / 'balanced_losses.csv'
        if epoch == 0:
            balanced_log_frame.to_csv(balanced_csv_path, index=False)
        else:
            balanced_log_frame.to_csv(balanced_csv_path, index=False, mode='a', header=False)
        
        blip_model.eval()
        accuracy_dict = evaluate_features(model=blip_model, dataset=valset, text_preprocessor=txt_processors["eval"], similarity_option=args.similarity_option)
        cur_acc = accuracy_dict['acc']
        if cur_acc > best_acc and args.save_training:
            best_acc = cur_acc
            logging.info('Save the current best model weights by mean average')
            torch.save(blip_model.state_dict(), training_path / 'best_model.pth')
        accuracy_list.append(accuracy_dict['acc'])
        

    with open('./res_acc.log', 'a+') as f:
        f.write(f'{args.timestamp}: {args.exp_name}\n')
        f.write(f'{str(accuracy_dict)}\n')

    return accuracy_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIRR')
    parser.add_argument('--method', type=str, default='image_diff')
    parser.add_argument('--backbone', type=str, default='pretrain')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=str, help='The index of used gpu', default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=30)
    
    parser.add_argument('--save_training', action='store_true', help='save model in training.')

    parser.add_argument('--inn_rho', type=float, default=0.8,
                        help='max marginal mass in UOT')
    parser.add_argument('--inn_sinkhorn_reg', type=float, default=0.05,
                        help='Sinkhorn entropy regularization')
    parser.add_argument('--inn_sinkhorn_iter', type=int, default=100,
                        help='max Sinkhorn iterations')
    parser.add_argument('--inn_negative_slope', type=float, default=0.05,
                        help='negative slope for LeakyReLU in InvertibleSingleLayer')

    parser.add_argument('--inn_block_dim', type=int, default=64,
                        help='block size for BlockOrthogonalFeatureTransform')
    parser.add_argument('--swap_warmup_epochs', type=int, default=5,
                        help='epochs for soft mixing of predicted and OT masks')
    parser.add_argument('--swap_coef_main', type=float, default=1.0,
                        help='coefficient for main swap loss')
    parser.add_argument('--swap_coef_aux1', type=float, default=0.5,
                        help='coefficient for aux swap loss 1')
    parser.add_argument('--swap_coef_aux2', type=float, default=0.5,
                        help='coefficient for aux swap loss 2')
    parser.add_argument('--swap_coef_bce', type=float, default=1.0,
                        help='coefficient for BCE mask prediction loss')

    parser.add_argument('--exp_name', type=str, default='')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    utility.set_device(args.gpu)
    set_seed(args.seed, args.shuffle_seed)
    if args.dataset.lower() == 'cirr':
        args.dataset = 'CIRR'
    elif args.dataset.lower() == 'fashioniq':
        args.dataset = 'FashionIQ'
    else:
        raise ValueError(f'The name of dataset {args.dataset} is invalid.')
    
    log_folder_path, timestamp = utility.get_log(args.dataset, args.exp_name)
    args.timestamp = timestamp
    file_name = './log/metrics.json'
    os.makedirs('./log', exist_ok=True)
    utility.Params.initialize(args)
    logging.info('Arguments:')
    for k in args.__dict__.keys():
        logging.info(f'    {k}:, {str(args.__dict__[k])}')

    save_path = os.path.join(log_folder_path, 'args.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(params(), f, indent=4, ensure_ascii=False)
    logging.info(f'Config saved to {save_path}')
    
    accuracy_list = blip_finetune(params)
    unique_key = f"{args.exp_name}-{timestamp}"
    this_dict = {unique_key:{'parameters': params(), 'accuracies': [round(num, 2) for num in accuracy_list], 
                            'max_acc':round(max(accuracy_list), 2), 'max_acc_epoch': int(np.argmax(accuracy_list)+1),
                            'last_epoch_acc':round(accuracy_list[-1], 2),
                }}
    if os.path.exists(file_name):
        with open(file_name, 'r') as json_file: 
            my_dict = json.load(json_file)
        my_dict.update(this_dict)
    else:
        my_dict = this_dict

    formatted_json_string = utility.custom_json_dumps(my_dict, indent=2)
    with open(file_name, 'w') as f:
        f.write(formatted_json_string)
