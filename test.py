from pickle import FALSE
import torch
import torch.nn as nn
import numpy as np
import utils
import os
import json
from train import *
from eval.eval_detection import ANETdetection

import pandas as pd
from tqdm import tqdm


def test(net, config, logger, test_loader, test_info, step, model_file=None,save_file=False):
    with torch.no_grad():
        net.eval()

        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}
        
        num_correct = 0.
        num_total = 0.

        load_iter = iter(test_loader)

        for i in tqdm(range(len(test_loader.dataset))):

            _, _data, _label, _, _, vid_name, vid_num_seg = next(load_iter)

            _data = _data.cuda()
            _label = _label.cuda()

            vid_num_seg = vid_num_seg[0].cpu().item()
            
            num_segments = _data.shape[1]

            vid_score, cas_sigmoid_fuse, _ = net(_data)
            ###################################
            if save_file:
                bkg_score=pd.DataFrame(cas_sigmoid_fuse[0,:,-1].cpu().data.numpy())
                save_dir=os.path.join(config.output_path,"saved_file","bkg_score")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                bkg_score.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)
            
            agnostic_score = 1 - cas_sigmoid_fuse[:,:,-1].unsqueeze(2)
            cas_sigmoid_fuse = cas_sigmoid_fuse[:,:,:-1]
            
            #video-level预测
            label_np = _label.cpu().data.numpy()
            score_np = vid_score[0].cpu().data.numpy()
            ###################################
            if save_file:
                score_pd=pd.DataFrame(score_np)
                save_dir=os.path.join(config.output_path,"saved_file","vid_score")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                score_pd.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)
                           
            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < config.class_thresh)] = 0
            pred_np[np.where(score_np >= config.class_thresh)] = 1
            if pred_np.sum() == 0:
                pred_np[np.argmax(score_np)] = 1

            correct_pred = np.sum(label_np == pred_np, axis=1)

            #计算正确率
            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]

            cas = cas_sigmoid_fuse
        
            pred = np.where(score_np >= config.class_thresh)[0]

            if len(pred) == 0:
                pred = np.array([np.argmax(score_np)])

            cas_pred = cas[0].cpu().numpy()[:, pred]
            ###################################
            if save_file:
                cas_pred_pd=pd.DataFrame(cas_pred)
                save_dir=os.path.join(config.output_path,"saved_file","cas_pred")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cas_pred_pd.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)
            cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
            
            #得到上采样的cas
            cas_pred = utils.upgrade_resolution(cas_pred, config.scale)
            ###################################
            if save_file:
                cas_pred_np=np.reshape(cas_pred,(cas_pred.shape[0],-1))
                cas_pred_pd=pd.DataFrame(cas_pred_np)
                save_dir=os.path.join(config.output_path,"saved_file","cas_pred_upgrade")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cas_pred_pd.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)
            
            proposal_dict = {}

            agnostic_score = agnostic_score.expand((-1, -1, config.num_classes))
            agnostic_score_np = agnostic_score[0].cpu().data.numpy()[:, pred]
            agnostic_score_np = np.reshape(agnostic_score_np, (num_segments, -1, 1))
            #得到上采样的bkg分数
            agnostic_score_np = utils.upgrade_resolution(agnostic_score_np, config.scale)
            ###################################
            if save_file:
                agnostic_score_save=np.reshape(agnostic_score_np,(agnostic_score_np.shape[0],-1))
                agnostic_score_pd=pd.DataFrame(agnostic_score_save)
                save_dir=os.path.join(config.output_path,"saved_file","agnostic_score_upgrade")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                agnostic_score_pd.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)
            

            for i in range(len(config.act_thresh_cas)):
                cas_temp = cas_pred.copy()

                zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh_cas[i])
                cas_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                vid_num_seg, config.feature_fps, num_segments)

                for i in range(len(proposals)):
                    class_id = proposals[i][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[i]

            for i in range(len(config.act_thresh_agnostic)):
                cas_temp = cas_pred.copy()

                agnostic_score_np_temp = agnostic_score_np.copy()

                zero_location = np.where(agnostic_score_np_temp[:, :, 0] < config.act_thresh_agnostic[i])
                agnostic_score_np_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(agnostic_score_np_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                vid_num_seg, config.feature_fps, num_segments)

                for i in range(len(proposals)):
                    class_id = proposals[i][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[i]

            final_proposals = []
            for class_id in proposal_dict.keys():
                final_proposals.append(utils.nms(proposal_dict[class_id], thresh=0.7))

            final_proposals = [final_proposals[i][j] for i in range(len(final_proposals)) for j in range(len(final_proposals[i]))]

            final_res['results'][vid_name[0]] = utils.result2json(final_proposals)

 

        test_acc = num_correct / num_total

        json_path = os.path.join(config.output_path, 'temp_result.json')
        with open(json_path, 'w') as f:
            json.dump(final_res, f)
            f.close()
        
        tIoU_thresh = np.linspace(0.1, 0.7, 7)
        anet_detection = ANETdetection(config.gt_path, json_path,
                                   subset='test', tiou_thresholds=tIoU_thresh,
                                   verbose=False, check_status=False)
        mAP, _ = anet_detection.evaluate()

        logger.log_value('acc/Test accuracy', test_acc, step)

        for i in range(tIoU_thresh.shape[0]):
            logger.log_value('acc/mAP@{:.1f}'.format(tIoU_thresh[i]), mAP[i], step)

        logger.log_value('acc/Average mAP[0.1:0.7]', mAP[:7].mean(), step)
        logger.log_value('acc/Average mAP[0.1:0.5]', mAP[:5].mean(), step)
        logger.log_value('acc/Average mAP[0.3:0.7]', mAP[2:7].mean(), step)

        test_info["step"].append(step)
        test_info["test_acc"].append(test_acc)
        test_info["average_mAP[0.1:0.7]"].append(mAP[:7].mean())
        test_info["average_mAP[0.1:0.5]"].append(mAP[:5].mean())
        test_info["average_mAP[0.3:0.7]"].append(mAP[2:7].mean())

        for i in range(tIoU_thresh.shape[0]):
            test_info["mAP@{:.1f}".format(tIoU_thresh[i])].append(mAP[i])

        

