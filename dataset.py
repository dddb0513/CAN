import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler
import numpy as np


# class HMERDataset(Dataset):
#     def __init__(self, params, image_path, label_path, words, is_train=True):
#         super(HMERDataset, self).__init__()
#         if image_path.endswith('.pkl'):
#             with open(image_path, 'rb') as f:
#                 self.images = pkl.load(f)
#         elif image_path.endswith('.list'):
#             with open(image_path, 'r') as f:
#                 lines = f.readlines()
#             self.images = {}
#             print(f'data files: {lines}')
#             for line in lines:
#                 name = line.strip()
#                 print(f'loading data file: {name}')
#                 start = time.time()
#                 with open(name, 'rb') as f:
#                     images = pkl.load(f)
#                 self.images.update(images)
#                 print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

#         with open(label_path, 'r') as f:
#             self.labels = f.readlines()

#         self.words = words
#         self.is_train = is_train
#         self.params = params

#     def __len__(self):
#         assert len(self.images) == len(self.labels)
#         return len(self.labels)

#     def __getitem__(self, idx):
#         name, *labels = self.labels[idx].strip().split()
#         name = name.split('.')[0] if name.endswith('jpg') else name
#         image = self.images[name]
#         image = torch.Tensor(255-image) / 255
#         image = image.unsqueeze(0)
#         labels.append('eos')
#         words = self.words.encode(labels)
#         words = torch.LongTensor(words)
#         return image, words

class MyDataset(Dataset):
    def __init__(self, params, file_paths, words, is_train=True):
        super(MyDataset, self).__init__()
        
        # 文件是一个 .npl 格式，包含字典列表
        # 例如，你可以使用 np.load 来读取这些文件
        # 如果文件是一个 pickle 格式，你可以使用 pkl.load
        # with open(file_path, 'rb') as f:
        #     self.data = np.load(f, allow_pickle=True)
        # 初始化时接收多个文件路径，并将它们的数据合并
        self.data = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                data = np.load(f, allow_pickle=True)
                self.data.extend(data)  # 将每个文件的数据添加到列表中
        
        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像和标签
        item = self.data[idx]
        image = item['image']
        labels = item['label'].split()  # 假设标签是一个字符串，可能是多个单词，需要分割
        
        # 处理图像
        image = torch.Tensor(255 - image) / 255  # 归一化到 [0, 1]
        image = image.permute(2, 0, 1)  # 转换为 (C, H, W) 格式
        
        # 将标签编码为对应的数字索引（假设 Words 类有编码方法）
        labels.append('eos')  # 添加结束标记
        word_indices = self.words.encode(labels)  # 使用你的编码方式
        word_indices = torch.LongTensor(word_indices)

        return image, word_indices


# def get_crohme_dataset(params):
#     words = Words(params['word_path'])
#     params['word_num'] = len(words)
#     print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
#     print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

#     train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
#     eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

#     train_sampler = RandomSampler(train_dataset)
#     eval_sampler = RandomSampler(eval_dataset)

#     train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
#                               num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
#     eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
#                               num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

#     print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
#           f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
#     return train_loader, eval_loader

def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    
    print(f"训练数据路径 : {params['train_file_path']} ")
    print(f"验证数据路径 : {params['eval_file_path']} ")

    # 使用 MyDataset 类
    train_dataset = MyDataset(params, params['train_file_path'], words, is_train=True)
    eval_dataset = MyDataset(params, params['eval_file_path'], words, is_train=False)

    # 创建数据加载器
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader

def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}
