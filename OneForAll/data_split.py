import os
import shutil


root = 'OneForAll/data/datasets'

if os.path.exists(os.path.join(root, 'vehicle')):
    shutil.rmtree(os.path.join(root, 'vehicle'))
if os.path.exists(os.path.join(root, 'pedestrian')):
    shutil.rmtree(os.path.join(root, 'pedestrian'))

for dataset_type in ['train', 'test', 'val']:
    os.makedirs(os.path.join(root, 'vehicle', dataset_type, f'{dataset_type}_images'))
    os.makedirs(os.path.join(root, 'pedestrian', dataset_type, f'{dataset_type}_images'))
    filename = f'{dataset_type}_text.txt' if dataset_type == 'test' else f'{dataset_type}_label.txt'
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
                shutil.copyfile(os.path.join(root, dataset_type, f'{dataset_type}_images', file), os.path.join(root, 'pedestrian', dataset_type, f'{dataset_type}_images', file))
            else:
                shutil.copyfile(os.path.join(root, dataset_type, f'{dataset_type}_images', file), os.path.join(root, 'vehicle', dataset_type, f'{dataset_type}_images', file))

# check
for data_type in ['vehicle', 'pedestrian']:
    for dataset_type in ['train', 'val']:
        filename = f'{dataset_type}_label.txt'
        data_file_set = set(os.listdir(os.path.join(root, data_type, dataset_type, f'{dataset_type}_images')))
        label_file_set = set()
        with open(os.path.join(root, data_type, dataset_type, filename), 'r') as f:
            for line in f:
                file = line.split('$', 1)[0]
                label_file_set.add(file)
        # print(data_file_set - label_file_set)
        # print(label_file_set - data_file_set)
        assert label_file_set.issubset(data_file_set)
        print(f'{data_type} {dataset_type} is checked')
