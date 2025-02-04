import os
import Augmentor
import shutil
import json
import random
import stanza
import re
import itertools
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from threading import Thread


def ready_for_aug(src_path: str, dst_path: str):
    assert src_path.endswith('train_images_raw')
    assert os.path.exists(src_path)
    os.makedirs(dst_path, exist_ok=True)
    for file in tqdm(os.listdir(src_path)):
        assert file.endswith('.jpg')
        if not os.path.exists(os.path.join(dst_path, file)):
            Image.open(os.path.join(src_path, file)).convert('RGB').save(os.path.join(dst_path, file))
    if os.path.exists(os.path.join(dst_path, 'output')):
        shutil.rmtree(os.path.join(dst_path, 'output'))
    p = Augmentor.Pipeline(dst_path)
    return p


def move_aug_result(src_path: str, dst_path: str, save_path: str):
    aug_res = defaultdict(list)
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path)
    aug_path = os.path.join(src_path, 'output')
    for file in os.listdir(aug_path):
        src = os.path.join(aug_path, file)
        dst = os.path.join(dst_path, file)
        shutil.move(src, dst)
        aug_res[file.split('_')[4]].append(file)
    shutil.rmtree(aug_path)
    for file in os.listdir(src_path):
        src = os.path.join(src_path, file)
        dst = os.path.join(dst_path, file)
        shutil.copyfile(src, dst)
        aug_res[file].append(file)
    with open(save_path, 'w') as f:
        json.dump(aug_res, f, indent=4)


def aug_vehicle_image():
    save_path = 'OneForAll/data/datasets/vehicle/train/aug_image.json'
    raw_path = 'OneForAll/data/datasets/vehicle/train/train_images_raw'
    clean_path = 'OneForAll/data/datasets/vehicle/train/train_images_clean'
    aug_path = 'OneForAll/data/datasets/vehicle/train/train_images'
    p = ready_for_aug(raw_path, clean_path)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.random_erasing(probability=0.3, rectangle_area=0.1)
    p.sample(3 * len(os.listdir(clean_path)))
    move_aug_result(clean_path, aug_path, save_path)


def aug_pedestrian_image():
    save_path = 'OneForAll/data/datasets/pedestrian/train/aug_image.json'
    raw_path = 'OneForAll/data/datasets/pedestrian/train/train_images_raw'
    clean_path = 'OneForAll/data/datasets/pedestrian/train/train_images_clean'
    aug_path = 'OneForAll/data/datasets/pedestrian/train/train_images'
    p = ready_for_aug(raw_path, clean_path)
    p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.random_erasing(probability=0.3, rectangle_area=0.1)
    p.sample(3 * len(os.listdir(clean_path)))
    move_aug_result(clean_path, aug_path, save_path)


def aug_vehicle_text():
    save_path = 'OneForAll/data/datasets/vehicle/train/aug_text.json'
    src_path = 'OneForAll/data/datasets/vehicle/train/train_label_raw.txt'
    aug_res = defaultdict(list)
    with open(src_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip().rsplit('$', 1)
            if line[1] in aug_res:
                assert len(aug_res[line[1]]) == 4
                continue
            aug_res[line[1]].append(line[1])
            if line[1].startswith('This is '):
                for flag in [0, 1, 2]:
                    if flag == 0:
                        description = line[1].split(' ', 3)[-1]
                    elif flag == 1:
                        description = line[1].split(' ', 2)[-1]
                    else:
                        description = 'This ' + line[1].split(' ', 3)[-1]
                    description = description[0].upper() + description[1:].lower()
                    aug_res[line[1]].append(description)
            elif line[1].startswith('A ') or line[1].startswith('An '):
                for flag in [0, 1, 2]:
                    if flag == 0:
                        description = line[1].split(' ', 1)[-1]
                    elif flag == 1:
                        description = 'This ' + line[1].split(' ', 1)[-1]
                    else:
                        description = 'This is ' + line[1]
                    description = description[0].upper() + description[1:].lower()
                    aug_res[line[1]].append(description)
            else:
                print(line[1])
                raise ValueError
    with open(save_path, 'w') as f:
        json.dump(aug_res, f, indent=4)


def process(lines, pos_path, save_path, thread_idx):
    aug_res = defaultdict(list)
    pos_res = defaultdict(str)
    pos = stanza.Pipeline('en', processors='tokenize,pos', download_method=None, use_gpu=False)
    keep_type = set({'NOUN', 'ADJ', 'NUM', 'SYM', 'PUNCT', 'VERB', 'ADP', 'ADV', 'PROPN'})
    for line in tqdm(lines, desc=f'thread {thread_idx}'):
        line = line.strip().rsplit('$', 1)
        if line[1] in aug_res:
            continue
        candidate = []
        for sent in pos(line[1]).to_dict():
            for word in sent:
                if word['upos'] not in keep_type:
                    candidate.append((word['start_char'], word['end_char']))
                pos_res[line[1]] += word['text'] + '|' + word['upos'] + ' '
        if len(candidate) < 5:
            delete_lists = (candidate,)
        else:
            delete_lists = itertools.combinations(candidate, 5)
        descriptions = []
        for delete_list in delete_lists:
            delete_list = sorted(delete_list, key=lambda x: x[0])
            description = line[1][:delete_list[0][0]]
            for i in range(1, len(delete_list)):
                description += line[1][delete_list[i-1][1]:delete_list[i][0]]
            description += line[1][delete_list[-1][1]:]
            description = re.sub(r'\s+', ' ', description)
            descriptions.append(description)
        aug_res[line[1]].append(line[1])
        for description in descriptions:
            aug_res[line[1]].append(description.strip())
    with open(save_path, 'w') as f:
        json.dump(aug_res, f, indent=4)
    with open(pos_path, 'w') as f:
        json.dump(pos_res, f, indent=4)


def aug_pedestrian_text():
    num_thread = 8
    save_path = 'OneForAll/data/datasets/pedestrian/train/aug_text.json'
    src_path = 'OneForAll/data/datasets/pedestrian/train/train_label_raw.txt'
    pos_path = 'OneForAll/data/datasets/pedestrian/train/train_text_pos.json'
    import subprocess
    num_sample = int(subprocess.getoutput(f"wc -l {src_path}").split()[0])
    num_sample_per_t = num_sample // num_thread
    t_list = []
    t_samples = []
    t_idx = 0
    num_processed = 0
    with open(src_path, 'r') as f:
        for idx, line in enumerate(f):
            t_samples.append(line)
            if (len(t_samples) == num_sample_per_t and t_idx < num_thread-1) or idx == num_sample-1:
                t = Thread(target=process, args=(t_samples, pos_path + f'_{t_idx}.json', save_path + f'_{t_idx}.json', t_idx))
                t.start()
                t_list.append(t)
                num_processed += len(t_samples)
                t_samples = []
                t_idx += 1
    assert num_sample == num_processed
    for t in t_list:
        t.join()
    aug_res = dict()
    pos_res = dict()
    for t_idx in range(num_thread):
        with open(save_path + f'_{t_idx}.json', 'r') as f:
            t_aug_res = json.load(f)
            aug_res = {**aug_res, **t_aug_res}
        with open(pos_path + f'_{t_idx}.json', 'r') as f:
            t_pos_res = json.load(f)
            pos_res = {**pos_res, **t_pos_res}
        os.remove(save_path + f'_{t_idx}.json')
        os.remove(pos_path + f'_{t_idx}.json')
    with open(save_path, 'w') as f:
        json.dump(aug_res, f, indent=4)
    with open(pos_path, 'w') as f:
        json.dump(pos_res, f, indent=4)


def merge_aug_result(data_type, config):
    label_path = f'OneForAll/data/datasets/{data_type}/train/train_label_raw.txt'
    save_path = f'OneForAll/data/datasets/{data_type}/train/train_label.txt'
    aug_image_path = f'OneForAll/data/datasets/{data_type}/train/aug_image.json'
    aug_text_path = f'OneForAll/data/datasets/{data_type}/train/aug_text.json'
    aug_image_dict = json.load(open(aug_image_path, 'r'))
    aug_text_dict = json.load(open(aug_text_path, 'r'))
    # raw
    num_image = config['num_image'] - 1
    num_text = config['num_text'] - 1
    with open(label_path, 'r') as f, open(save_path, 'w') as f_w:
        for line in tqdm(f):
            line = line.strip().split('$')
            selected_image = []
            selected_text = []
            # raw
            selected_image.append(aug_image_dict[line[0]][-1])
            selected_text.append(aug_text_dict[line[2]][0])
            selected_image += random.sample(aug_image_dict[line[0]][:-1], k=min(len(aug_image_dict[line[0]][:-1]), num_image))
            selected_text += random.sample(aug_text_dict[line[2]][1:], k=min(len(aug_text_dict[line[2]][1:]), num_text))
            for aug_text in selected_text:
                f_w.write(selected_image[0] + '$' + line[1] + '$' + aug_text + '\n')
            for aug_image in selected_image[1:]:
                f_w.write(aug_image + '$' + line[1] + '$' + selected_text[0] + '\n')


if __name__ == '__main__':
    # 数据增强时，打开以下注释
    # aug_vehicle_image()
    # aug_pedestrian_image()
    # aug_vehicle_text()
    # aug_pedestrian_text()

    # 合并数据增强结果
    config = {
        'vehicle': {
            'image': True,
            'num_image': 3,
            'text': True,
            'num_text': 2
        },
        'pedestrian': {
            'image': True,
            'num_image': 3,
            'text': True,
            'num_text': 2
        }
    }
    for data_type in ['vehicle', 'pedestrian']:
        merge_aug_result(data_type, config[data_type])
