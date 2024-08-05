from cnn_lstm import *
from Dataloader import *
from models import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datetime import datetime

def plot_output_vs_label(outputs, labels):

    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    plt.figure()
    plt.plot(labels[:, 0], color='red', label='ground truth, dx')
    plt.plot(outputs[:, 0], linestyle='--', color='red', label='test, dx')
    plt.plot(labels[:, 1], color='green', label='ground truth, dy')
    plt.plot(outputs[:, 1], linestyle='--', color='green', label='test, dy')
    plt.plot(labels[:, 2], color='blue', label='ground truth, dz')
    plt.plot(outputs[:, 2], linestyle='--', color='blue', label='test, dz')
    plt.legend()
    plt.ylabel('positions')
    plt.xlabel('')
    plt.show()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)

            batch_size = images.size(0)
            h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
            c0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

            outputs, _ = model(images, (h0, c0))
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(dataloader)

def train(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device):

    best_loss = float('inf')
    best_model_path = './model/best_model'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.pth'
    final_model_path = './model/final_model'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.pth'


    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    hidden = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, is_new_sequence in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            if is_new_sequence.any():
                hidden = None  # Reset hidden state for new sequences
            #print(images.size())
            outputs, hidden = model(images, hidden)
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with validation loss: {best_loss:.4f}')
    # save the model
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved after {num_epochs} epochs')

def test(model, dataloader, device, criterion=None):
    model.eval()

    all_outputs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            labels = labels.squeeze(1)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    #print(all_labels)
    if criterion is not None:
        avg_loss = total_loss / len(dataloader)
        print(f'Average Loss: {avg_loss:.4f}')

    plot_output_vs_label(all_outputs, all_labels)

if __name__ == '__main__':

    image_dirs = ['./data/images/exp2', './data/images/exp3', './data/images/exp4']
    label_files = ['./data/state/exp2/state.txt', './data/state/exp3/state.txt', './data/state/exp4/state.txt']
    test_image_dir = ['./data/images/exp1']
    test_label_file = ['./data/state/exp1/state.txt']
    model_file = './model/best_model20240709_154140.pth'
    sequence_length = 3  # 每个小 sequence 的长度

    batch_size = 8
    num_epochs = 30
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_dataset = MultiSequenceDataset(image_dirs, label_files, sequence_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = MultiSequenceDataset(test_image_dir, test_label_file, sequence_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    conv_output = 16
    hidden_size = 128
    num_layers = 2
    output_size = 3
    model = CNN_LSTM(conv_output, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_file))

    #train(model, train_dataloader, test_dataloader, num_epochs, learning_rate, device)
    test(model, test_dataloader, device, nn.MSELoss())










