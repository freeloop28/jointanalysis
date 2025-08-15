from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from lib.tracker.multitracker import JDETracker
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
from collections import defaultdict

from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.sperm_cluster import cluster_tracks_and_map_grades, extract_features_for_clustering
from opts import opts
import joblib

gmm = joblib.load('gmm_model.joblib')
label_to_grade = joblib.load('label_to_grade.joblib')

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                track_id_out = track_id
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id_out, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                track_id_out = track_id
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id_out, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    all_tracks = {}   # 新增：收集轨迹点
    all_tracks_with_frame = {} 
    img_paths = {}    # 新增：收集每帧图片路径
    img0s = {}        # 存每帧原图（不区分图片/视频模式都能用）

    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, data in enumerate(dataloader):
        #if i % 8 != 0:
            #continue

        if isinstance(data[0], str):  # 图片集模式：img_path, img, img0
            img_path, img, img0 = data
            current_frame_id = i + 1
        else:  # 视频模式：frame_id_in_video, img, img0
            frame_id_in_video, img, img0 = data
            img_path = None
            current_frame_id = frame_id_in_video + 1  

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
                # 收集轨迹点
                x, y, w, h = tlwh   #收集轨迹点
                center = (x + w/2, y + h/2)  #收集轨迹点
                all_tracks.setdefault(tid, []).append(center)  # 原格式（聚类用）
                all_tracks_with_frame.setdefault(tid, []).append((frame_id + 1, center))  # 可视化
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #img_paths[frame_id + 1] = img_path   # 保存每帧图片路径
        img0s[current_frame_id] = img0.copy()  # 不管是图片还是视频，都直接存img0（保证后续可用）
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        # if show_image or save_dir is not None:
        #     online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
        #                                   fps=1. / timer.average_time)
        # if show_image:
        #     cv2.imshow('online_im', online_im)
        # if save_dir is not None:
        #     cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    #return frame_id, timer.average_time, timer.calls
    return results, all_tracks, all_tracks_with_frame, img0s, timer.average_time, timer.calls

def draw_all_grades_trajectories_on_video(
    output_path,
    all_tracks_with_frame,
    track_id_to_grade,
    img0s,
    grade_to_color = {
        'A': (255, 0, 0),    # 蓝色
        'B': (0, 255, 0),  # 绿色
        'C': (0, 255, 255),    # 黄色
        'D': (0, 0, 255),    # 红色
        'U': (128, 128, 128) # 未知/未分级
    }
):
    # 按照grade分类ID
    grade_to_ids = {}
    for tid, grade in track_id_to_grade.items():
        grade_to_ids.setdefault(grade, []).append(tid)

    # 每个grade一组轨迹
    grade_tracks = {}
    for grade, ids in grade_to_ids.items():
        tracks = {}
        for tid in ids:
            if tid in all_tracks_with_frame:
                tracks[tid] = all_tracks_with_frame[tid]
        grade_tracks[grade] = tracks

    img_keys = sorted(img0s)
    img_list = [img0s[k] for k in img_keys]
    h, w = img_list[0].shape[:2]
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 为每个grade和ID分别存历史轨迹
    history_points = {grade: {tid: [] for tid in tracks} for grade, tracks in grade_tracks.items()}

    for idx, frame_num in enumerate(img_keys):
        frame = img0s[frame_num].copy()
        for grade, tracks in grade_tracks.items():
            color = grade_to_color.get(grade, (128,128,128))
            for tid, points in tracks.items():
                # 更新历史轨迹
                for f_id, pt in points:
                    if f_id == frame_num:
                        history_points[grade][tid].append((int(pt[0]), int(pt[1])))

                # 画轨迹线
                pts = history_points[grade][tid]
                if len(pts) > 1:
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i-1], pts[i], color, 2)
                # 画编号
                if len(pts) > 0:
                    text = str(tid)
                    last_point = pts[-1]
                    cv2.putText(
                        frame, text, (last_point[0]+5, last_point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                    )
        out.write(frame)
    out.release()
    print('[Info] 所有轨迹连线视频已保存:', output_path)


def predict_grade_for_tracks(tracks, gmm, label_to_grade, speed_weight=40.0, curvature_weight=0.01, pad_length=60):
    if len(tracks) == 0:
        return []
    features = extract_features_for_clustering(tracks, speed_weight, curvature_weight, pad_length)
    labels = gmm.predict(features)
    grades = [label_to_grade.get(label, "U") for label in labels]
    return grades

def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # 强制清空 input_video，避免跳进视频分支
    opt.input_video = None
    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    if hasattr(opt, 'input_video') and opt.input_video is not None:
        logger.info(f'Running on video: {opt.input_video}')
        dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
        # 你可以自定义 video 名字作为 seq，比如 "video"
        seq = os.path.splitext(os.path.basename(opt.input_video))[0]
        output_dir = os.path.join(result_root, seq)
        mkdir_if_missing(output_dir)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        frame_rate = 25  # 你可以写死，或者用 opencv 读视频帧率
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                                save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)
        # 若要保存视频帧可以用 ffmpeg 合成
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
        # 不走 MOT Challenge 评测，直接返回
        return



    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        if output_dir is not None:
            mkdir_if_missing(output_dir)

        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        # nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
        #                       save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        # n_frame += nf
        # timer_avgs.append(ta)
        # timer_calls.append(tc)
            # ⭕ 调用新版eval_seq：返回results, all_tracks, img_paths, ta, tc
        results, all_tracks, all_tracks_with_frame, img0s, ta, tc = eval_seq(
            opt, dataloader, data_type, result_filename,
            frame_rate=frame_rate, use_cuda=True
        )
        n_frame += len(results)
        timer_avgs.append(ta)
        timer_calls.append(tc)

                # 1. 得到每个id的轨迹长度
        valid_ids = set()
        for tid, points in all_tracks.items():
            if len(points) >= 60:  # 轨迹完整（你也可以灵活改成别的阈值）
                valid_ids.add(tid)

        print(f"[DEBUG] 找到 {len(valid_ids)} 条长度 ≥60 的有效轨迹")

        print("所有track长度分布：")
        for tid, points in all_tracks.items():
            print(f"track_id={tid}, len={len(points)}")


        # ⭕ 轨迹聚类和分级
        valid_ids = set([tid for tid, pts in all_tracks.items() if len(pts) >= 60])
        track_ids = [tid for tid in all_tracks if tid in valid_ids]
        tracks = [all_tracks[tid] for tid in track_ids]

        if len(tracks) == 0:
            print(f"[ERROR] seq={seq}，无足够轨迹进行聚类，跳过本序列。")
            continue

        grades = predict_grade_for_tracks(tracks, gmm, label_to_grade)
        track_id_to_grade = {tid: grade for tid, grade in zip(track_ids, grades)}
        blue_ids = [tid for tid, grade in track_id_to_grade.items() if grade == 'A']
        blue_ids_path = os.path.join(result_root, f"{seq}_blue_ids.joblib")
        joblib.dump(blue_ids, blue_ids_path)
        print(f"[INFO] 已保存蓝色（Grade A）ID到: {blue_ids_path}")
        # 如果没有有效轨迹就跳过本 sequence，不要跑空数据聚类
        if len(tracks) == 0:
            print(f"[ERROR] seq={seq}，无足够轨迹进行聚类，跳过本序列。")
            continue
        print(f"[DEBUG] seq={seq}, len(valid_ids)={len(valid_ids)}, len(track_ids)={len(track_ids)}, len(tracks)={len(tracks)}")
        if len(tracks) == 0:
            print("[ERROR] No tracks for clustering after length filter!")
            continue

        # ⭕ 逐帧重新可视化并保存
        for frame_id, tlwhs, track_ids_in_frame in results:
            #img = cv2.imread(img_paths[frame_id])
            if output_dir is not None:
                img = img0s[frame_id]
                        # 只保留valid_id的目标
                filtered_tlwhs = []
                filtered_obj_ids = []
                for tlwh, obj_id in zip(tlwhs, track_ids_in_frame):
                    if obj_id in valid_ids:
                        filtered_tlwhs.append(tlwh)
                        filtered_obj_ids.append(obj_id)
                online_im = vis.plot_tracking(
                    img, filtered_tlwhs, filtered_obj_ids, track_ids_in_frame, frame_id=frame_id,
                    track_id_to_grade=track_id_to_grade
                )
                cv2.imwrite(os.path.join(output_dir, '{:05d}.jpg'.format(frame_id)), online_im)
                # ⭕ 保存蓝色精子轨迹连线视频

        all_grades_output_video_path = osp.join(output_dir, '{}_all_grades.mp4'.format(seq))
        draw_all_grades_trajectories_on_video(
            all_grades_output_video_path,
            all_tracks_with_frame,
            track_id_to_grade,
            img0s,
        )

        # 可选：视频合成
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = f'ffmpeg -f image2 -i {output_dir}/%05d.jpg -c:v libx264 {output_video_path}'
            os.system(cmd_str)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    # 覆盖掉 task，防止被命令行参数影响
    opt.task = 'mot'
    # print('label_to_grade:', label_to_grade)
    # print('GMM n_components:', gmm.n_components)

    opt.load_model = '/home/ubuntu/projects/FairMOT/exp/mot/sperm_exp/model_30.pth'
    # 手动指定你要跑的序列
    data_root = '/home/ubuntu/projects/FairMOT/minkidataset'
    seqs = ['017']   # 你的精子实验序列编号

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='sperm_track_test',
         show_image=False,
         save_images=True,
         save_videos=True)



