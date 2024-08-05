from cnn_lstm import *
import torch
from cvxopt import solvers, matrix
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

class DepthImageDataset(Dataset):
    def __init__(self, image_dir, label_file):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.labels = self.load_labels(label_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)

    def load_labels(self, label_file):
        labels = []
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                #file_name = parts[0]
                label = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
                labels.append(label)
        return labels


class MultiSequenceDataset(Dataset):
    def __init__(self, image_dirs, state_dirs, sequence_length, transform=None):
        self.image_dirs = image_dirs
        self.state_dirs = state_dirs
        self.transform = transform
        self.sequence_length = sequence_length
        #self.sequences = []
        self.data = self.load_data()

        # Load all sequences
        #for img_dir, lbl_file in zip(image_dirs, label_files):
        #    images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        #    labels = np.loadtxt(lbl_file)
        #    self.sequences.append((images, labels))

    def __len__(self):
        #return sum(len(labels) for _, labels in self.sequences)
        return len(self.data)

    def __getitem__(self, idx):
        # Determine which sequence the idx belongs to
        '''
        current_idx = 0
        for images, labels in self.sequences:
            if idx < current_idx + len(labels):
                seq_idx = idx - current_idx
                image_path = images[seq_idx]
                label = labels[seq_idx]
                label = np.array([float(label[0]), float(label[1]), float(label[2])], dtype=np.float32)

                is_new_sequence = seq_idx == 0  # If this is the first element of the sequence
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (128, 128))
                image = image.astype(np.float32) / 255.0
                image = np.expand_dims(image, axis=0)

                #image = Image.open(image_path).convert('L')
                #if self.transform:
                #    image = self.transform(image)

                # Convert to tensor
                #image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                #label = torch.tensor(label, dtype=torch.float32)

                return torch.tensor(image), torch.tensor(label), is_new_sequence

            current_idx += len(labels)

        raise IndexError(f'Index {idx} out of range')

        '''
        images = []
        states = []
        image_sequence, state_sequence, is_new_sequences = self.data[idx]
        for image in image_sequence:
            #images = [torch.tensor(np.load(image)).float() ]
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            images.append(image)

        #for state in state_sequence:
        #    state = np.array([float(state[0]), float(state[1]), float(state[2])], dtype=np.float32)
        #    states.append(state)
        state = state_sequence
        states = [np.array([float(state[0]), float(state[1]), float(state[2])], dtype=np.float32)]

        return torch.tensor(images), torch.tensor(states), is_new_sequences

    def load_data(self):
        all_sequences = []

        for image_dir, state_dir in zip(self.image_dirs, self.state_dirs):
            images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
            states = np.loadtxt(state_dir)

            # Ensure the number of images matches the number of state entries
            assert len(images) == len(states), "Mismatch between images and states count"

            for i in range(0, len(images) - self.sequence_length + 1, self.sequence_length):
                is_new_sequence = False
                if i == 0:
                    is_new_sequence = True
                image_sequence = images[i:i + self.sequence_length]
                #state_sequence = states[i:i + self.sequence_length]
                state = states[i + self.sequence_length - 1]
                #state_sequence = np.array([float(states[i:i + self.sequence_length])], dtype=np.float32)

                all_sequences.append((image_sequence, state, is_new_sequence))

        return all_sequences

class BNDataset(Dataset):
    def __init__(self, state_dir, cmd_dir):

        self.states, self.labels = self.load_labels(state_dir, cmd_dir)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        state = self.states[idx]
        label = self.labels[idx]
        return torch.tensor(state, dtype=torch.double), torch.tensor(label, dtype=torch.double)

    def get_mean_std(self):
        states_np = np.array(self.states)
        labels_np = np.array(self.labels)

        mean_states = np.mean(states_np, axis=0)
        std_states = np.std(states_np, axis=0)

        mean_labels = np.mean(labels_np, axis=0)
        std_labels = np.std(labels_np, axis=0)

        return (mean_states, std_states), (mean_labels, std_labels)

    def load_labels(self, state_dir, cmd_dir):
        cmds = []
        states = []
        with open(state_dir, 'r') as file:
            for line in file:
                parts = line.strip().split()
                label = np.array([float(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4]), float(parts[5])
                                  #float(parts[6]), float(parts[7]), float(parts[8])
                                  ], dtype=np.float32)
                states.append(label)

        with open(cmd_dir, 'r') as file:
            for line in file:
                parts = line.strip().split()
                label = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
                cmds.append(label)
        return states, cmds

def test_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)

    return sol['x']



# Example usage:
'''
image_dirs = ['./data/images/exp1', './data/images/exp2']
#cmd_files = './data/state/exp1/cmd.txt'
state_dirs = ['./data/state/exp1/state.txt', './data/state/exp2/state.txt']
sequence_length = 5  # 每个小 sequence 的长度

dataset = MultiSequenceDataset(image_dirs, state_dirs, sequence_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

# 测试 DataLoader
for images, state, _ in dataloader:
    print(len(images), len(state))
'''