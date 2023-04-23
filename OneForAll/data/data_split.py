import os
import shutil


root = 'OneForAll/data/datasets'

if os.path.exists(os.path.join(root, 'vehicle')):
    shutil.rmtree(os.path.join(root, 'vehicle'))
if os.path.exists(os.path.join(root, 'pedestrian')):
    shutil.rmtree(os.path.join(root, 'pedestrian'))


for dataset_type in ['train', 'test', 'val']:
    if dataset_type == 'train':
        v_image_dir = os.path.join(root, 'vehicle', dataset_type, f'{dataset_type}_images_raw')
        p_image_dir = os.path.join(root, 'pedestrian', dataset_type, f'{dataset_type}_images_raw')
    else:
        v_image_dir = os.path.join(root, 'vehicle', dataset_type, f'{dataset_type}_images')
        p_image_dir = os.path.join(root, 'pedestrian', dataset_type, f'{dataset_type}_images')
    os.makedirs(v_image_dir)
    os.makedirs(p_image_dir)
    if dataset_type == 'test':
        filename = f'{dataset_type}_text.txt'
    elif dataset_type == 'val':
        filename = f'{dataset_type}_label.txt'
    elif dataset_type == 'train':
        filename = f'{dataset_type}_label_raw.txt'
    with open(os.path.join(root, dataset_type, filename), 'r') as f, \
        open(os.path.join(root, 'vehicle', dataset_type, filename), 'w') as f_v, \
        open(os.path.join(root, 'pedestrian', dataset_type, filename), 'w') as f_p:
        for line in f:
            if len(line) < 80:
                f_v.write(line)
            else:
                f_p.write(line)
        for file in os.listdir(os.path.join(root, dataset_type, f'{dataset_type}_images')):
            assert os.path.isfile(os.path.join(root, dataset_type, f'{dataset_type}_images', file))
            if file[:-4].isdigit():
                shutil.copyfile(os.path.join(root, dataset_type, f'{dataset_type}_images', file), os.path.join(p_image_dir, file))
            else:
                shutil.copyfile(os.path.join(root, dataset_type, f'{dataset_type}_images', file), os.path.join(v_image_dir, file))

# check
for data_type in ['vehicle', 'pedestrian']:
    for dataset_type in ['train', 'val']:
        filename = f'{dataset_type}_label_raw.txt' if dataset_type == 'train' else f'{dataset_type}_label.txt'
        dir_name = f'{dataset_type}_images_raw' if dataset_type == 'train' else f'{dataset_type}_images'
        data_file_set = set(os.listdir(os.path.join(root, data_type, dataset_type, dir_name)))
        label_file_set = set()
        with open(os.path.join(root, data_type, dataset_type, filename), 'r') as f:
            for line in f:
                file = line.split('$', 1)[0]
                label_file_set.add(file)
        # print(data_file_set - label_file_set)
        # print(label_file_set - data_file_set)
        assert label_file_set.issubset(data_file_set)
        print(f'{data_type} {dataset_type} is checked')
