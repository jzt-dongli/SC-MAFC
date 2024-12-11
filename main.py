import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve ,auc
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

def smooth_predictions(scores, sigma=3):
    smoothed_scores = gaussian_filter1d(scores, sigma=sigma)
    return smoothed_scores

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, [x1, x2, x3, x4]

    def decoder(self, x5, encoder_features):
        x1, x2, x3, x4 = encoder_features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def forward(self, x, x5_fused=None, encoder_features=None):
        if x5_fused is None or encoder_features is None:
            x5, encoder_features = self.encoder(x)
            logits = self.decoder(x5, encoder_features)
            return logits, x5
        else:
            logits = self.decoder(x5_fused, encoder_features)
            return logits

class TripleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(TripleUNet, self).__init__()
        self.unet_app = UNet(n_channels * 3, n_classes * 3, bilinear)
        self.unet_flow = UNet(n_channels * 2, n_classes * 2, bilinear)
        self.unet_pose = UNet(n_channels * 3, n_classes * 3, bilinear)

        self.flow_memory = Memory(feature_dim=512, key_dim=512, memory_size=10).to(device)
        self.pose_memory = Memory(feature_dim=512, key_dim=512, memory_size=10).to(device)

        self.app_fusion_conv = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.x_fusion_conv = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.final_fusion_conv = nn.Sequential(
            nn.Conv2d(512 * 2, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.space_conv1 = nn.Conv2d(512, 1, kernel_size=1)
        self.space_conv2 = nn.Conv2d(1, 512, kernel_size=1)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1024)

        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 32)

    def fusion(self, app, flow_or_pose):
        # channel_enhance
        x_avg_pool = F.adaptive_avg_pool2d(flow_or_pose, (1, 1))
        app_avg_pool = F.adaptive_avg_pool2d(app, (1, 1))
        x_max_pool = F.adaptive_max_pool2d(flow_or_pose, (1, 1))
        app_max_pool = F.adaptive_max_pool2d(app, (1, 1))

        concatenated_pools = torch.cat([x_avg_pool, x_max_pool, app_avg_pool, app_max_pool], dim=1)
        concatenated_pools = concatenated_pools.squeeze(-1).squeeze(-1)
        concatenated_pools = F.relu(self.fc1(concatenated_pools))
        mlp_concatenated_pools = self.fc2(concatenated_pools)

        splits = torch.split(mlp_concatenated_pools, 512, dim=1)
        x_pool_mlp, app_pool_mlp = [split.view(split.size(0), 512, 1, 1) for split in splits]

        app_weights = torch.sigmoid(app_pool_mlp)
        app_channel_enhance = app * app_weights

        x_weights = torch.sigmoid(x_pool_mlp)
        x_channel_enhance = flow_or_pose * x_weights

        # space_enhance
        app_channel_pool = self.space_conv1(app)
        x_channel_pool = self.space_conv1(flow_or_pose)
        app_flat = app_channel_pool.view(app_channel_pool.size(0), -1)
        x_flat = x_channel_pool.view(x_channel_pool.size(0), -1)
        concatenated_flat = torch.cat([x_flat, app_flat], dim=1)
        concatenated_flat = F.relu(self.fc3(concatenated_flat))
        mlp_concatenated_flat = self.fc4(concatenated_flat)

        app_restored = mlp_concatenated_flat[:, :16].view(-1, 1, 4, 4)
        x_restored = mlp_concatenated_flat[:, 16:].view(-1, 1, 4, 4)
        app_attention_map = self.space_conv2(torch.sigmoid(app_restored))
        x_attention_map = self.space_conv2(torch.sigmoid(x_restored))

        app_space_enhance = app * app_attention_map
        x_space_enhance = flow_or_pose * x_attention_map

        # app_fusion = 0.5 * app + 0.25 * x_channel_enhance + 0.25 * x_space_enhance
        # x_fusion = 0.5 * flow_or_pose + 0.25 * app_channel_enhance + 0.25 * app_space_enhance

        app_fusion_input = torch.cat([app, x_channel_enhance, x_space_enhance], dim=1)
        x_fusion_input = torch.cat([flow_or_pose, app_channel_enhance, app_space_enhance], dim=1)

        app_fusion = self.app_fusion_conv(app_fusion_input)
        x_fusion = self.x_fusion_conv(x_fusion_input)

        return app_fusion, x_fusion

    def forward(self, input_app, input_flow, input_pose):

        x5_app, enc_feat_app = self.unet_app.encoder(input_app)
        x5_flow, enc_feat_flow = self.unet_flow.encoder(input_flow)
        x5_pose, enc_feat_pose = self.unet_pose.encoder(input_pose)

        mem_x5_flow, flow_gathering_loss, flow_spreading_loss, flow_mem_reconstruction_loss = self.flow_memory(x5_flow)
        mem_x5_pose, pose_gathering_loss, pose_spreading_loss, pose_mem_reconstruction_loss = self.pose_memory(x5_pose)

        # fusion_input_app_flow = torch.cat((x5_app, x5_flow), dim=1)
        # x5_app_flow = self.fusion_app_flow(fusion_input_app_flow)
        x5_app_flow, x5_flow_app = self.fusion(x5_app, mem_x5_flow)

        # fusion_input_app_pose = torch.cat((x5_app, x5_pose), dim=1)
        # x5_app_pose = self.fusion_app_pose(fusion_input_app_pose)
        x5_app_pose, x5_pose_app = self.fusion(x5_app, mem_x5_pose)

        concat_app = torch.cat([x5_app_flow, x5_app_pose], dim=1)
        final_app = self.final_fusion_conv(concat_app)

        # logits_app_flow = self.unet_app.decoder(x5_app_flow, enc_feat_app)
        # logits_app_pose = self.unet_app.decoder(x5_app_pose, enc_feat_app)
        logits_app = self.unet_app.decoder(final_app, enc_feat_app)
        logits_flow_app = self.unet_flow.decoder(x5_flow_app, enc_feat_flow)
        logits_pose_app = self.unet_pose.decoder(x5_pose_app, enc_feat_pose)

        return logits_app, logits_flow_app, logits_pose_app, \
               flow_gathering_loss, flow_spreading_loss, flow_mem_reconstruction_loss, \
               pose_gathering_loss, pose_spreading_loss, pose_mem_reconstruction_loss

class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.m_items = Parameter(F.normalize(torch.rand((memory_size, key_dim), dtype=torch.float), dim=1),
                                 requires_grad=True).to(device)

    def get_score(self, mem, query):
        batch_size, height, width, channels = query.shape
        query_reshaped = query.reshape(-1, channels)  # Shape: (64 * 4 * 4, 512)

        score = torch.matmul(query_reshaped, mem.t())
        score = score.reshape(batch_size, height, width, -1)  # Shape: (64, 4, 4, 10)
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)

        return score_query, score_memory

    def forward(self, query):
        query = query.permute(0, 2, 3, 1)
        # gathering loss
        gathering_loss = self.gather_loss(query, self.m_items)
        # spreading_loss
        spreading_loss = self.spread_loss(query, self.m_items)
        # read
        updated_query, softmax_score_query, softmax_score_memory = self.read(query, self.m_items)
        mem_mse_loss = nn.MSELoss()
        mem_reconstruction_loss = mem_mse_loss(query, updated_query)
        updated_query = updated_query.permute(0, 3, 1, 2)
        return updated_query, gathering_loss, spreading_loss, mem_reconstruction_loss

    def spread_loss(self, query, keys, k=1):
        loss = torch.nn.TripletMarginLoss(margin=1.0)
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        _, gathering_indices = torch.topk(softmax_score_memory, k + 1, dim=3)

        # 1st, 2nd closest memories
        pos = keys[gathering_indices[:, :, :, 0]]
        neg = keys[gathering_indices[:, :, :, k]]

        spreading_loss = loss(query, pos, neg)

        return spreading_loss

    def gather_loss(self, query, keys):
        loss_fn = nn.MSELoss()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        topk_scores, gathering_indices = torch.topk(softmax_score_memory, 1, dim=3)

        gathering_indices = gathering_indices.squeeze(-1)

        gathered_keys = keys[gathering_indices]

        gather_loss = loss_fn(gathered_keys, query)

        return gather_loss

    def read(self, query, keys):
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        updated_query = torch.matmul(softmax_score_memory, keys)

        return updated_query, softmax_score_query, softmax_score_memory


def sort_files(file_path):
    directory, file_name = os.path.split(file_path)

    folder_number = int(os.path.basename(directory))

    file_parts = file_name.replace('.npy', '').split('_')
    file_primary, file_secondary = map(int, file_parts)

    return (folder_number, file_primary, file_secondary)

class VideoFrameDataset(Dataset):
    def __init__(self, file_paths, transform=None):

        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]
        sample = np.load(file_path)

        inputs = sample[:, :4, :, :].reshape(-1, 64, 64)
        labels = sample[:, 4, :, :]

        if self.transform:
            inputs = self.transform(inputs)
            labels = self.transform(labels)

        return inputs, labels

class test_VideoFrameDataset(Dataset):
    def __init__(self, full_frame_dir, obj_level_dir, transform=None):

        self.file_paths = []
        self.frame_ids = []
        self.transform = transform


        full_frame_folders = sorted(glob.glob(os.path.join(full_frame_dir, '*')))
        obj_level_folders = sorted(glob.glob(os.path.join(obj_level_dir, '*')))
        num_frames = 0

        for full_folder, obj_folder in zip(full_frame_folders, obj_level_folders):
            obj_files = sorted(glob.glob(os.path.join(obj_folder, '*.npy')))

            for obj_file in obj_files:
                frame_number = int(os.path.basename(obj_file).split('_')[0])
                global_frame_id = frame_number + num_frames
                self.file_paths.append(obj_file)

                self.frame_ids.append(global_frame_id)
            num_frames += len(glob.glob(os.path.join(full_folder, '*.jpg')))
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        frame_id = self.frame_ids[idx]
        sample = np.load(file_path)

        inputs = sample[:, :4, :, :].reshape(-1, 64, 64)
        labels = sample[:, 4, :, :].reshape(-1, 64, 64)

        if self.transform:
            inputs = self.transform(inputs)
            labels = self.transform(labels)

        return inputs, labels, frame_id

def pretrain_process(model, dataloader, attribute, dataset):
    num_epochs = 10
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader):

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs, _ = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            # print(f"{attribute} loss:", loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs},{attribute} Train Loss: {running_loss / len(dataloader)}')

    print(f'{attribute} Finished Training')

    torch.save(model.state_dict(), f'model_state/{dataset}_model_{attribute}_weights.pth')


def info_nce_loss(features1, features2, temperature=0.1):

    features1 = features1.view(features1.size(0), -1)
    features2 = features2.view(features2.size(0), -1)

    cos_sim = torch.mm(features1, features2.T) / temperature

    batch_size = features1.size(0)
    labels = torch.arange(batch_size).to(features1.device)

    loss = F.cross_entropy(cos_sim, labels)

    return loss


if __name__ == '__main__':


    is_shuffle = False
    dataset = 'shanghaitech'

    # # ped2
    # dataset_path_app = '/mnt/c/dataset/ped2/object_level/training/frames'
    # dataset_path_flow = '/mnt/c/dataset/ped2/object_level/training/flows'
    # dataset_path_pose = '/mnt/c/dataset/ped2/object_level/training/poses'
    #
    # full_frame_dir = '/mnt/c/dataset/avenue/testing/frames'
    # test_path_app = '/mnt/c/dataset/avenue/object_level/testing/frames'
    # test_path_flow = '/mnt/c/dataset/avenue/object_level/testing/flows'
    # test_path_pose = '/mnt/c/dataset/avenue/object_level/testing/poses'

    # # avenue
    # dataset_path_app = '/mnt/c/dataset/avenue/object_level/training/frames'
    # dataset_path_flow = '/mnt/c/dataset/avenue/object_level/training/flows'
    # dataset_path_pose = '/mnt/c/dataset/avenue/object_level/training/poses'
    #
    # full_frame_dir = '/mnt/c/dataset/avenue/testing/frames'
    # test_path_app = '/mnt/c/dataset/avenue/object_level/testing/frames'
    # test_path_flow = '/mnt/c/dataset/avenue/object_level/testing/flows'
    # test_path_pose = '/mnt/c/dataset/avenue/object_level/testing/poses'

    # shanghaitech
    dataset_path_app = '/mnt/c/dataset/shanghaitech/object_level/training/frames'
    dataset_path_flow = '/mnt/c/dataset/shanghaitech/object_level/training/flows'
    dataset_path_pose = '/mnt/c/dataset/shanghaitech/object_level/training/poses'

    full_frame_dir = '/mnt/c/dataset/shanghaitech/testing/frames'
    test_path_app = '/mnt/c/dataset/shanghaitech/object_level/testing/frames'
    test_path_flow = '/mnt/c/dataset/shanghaitech/object_level/testing/flows'
    test_path_pose = '/mnt/c/dataset/shanghaitech/object_level/testing/poses'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app_npy_files = glob.glob(os.path.join(dataset_path_app, '**/*.npy'), recursive=True)
    flow_npy_files = glob.glob(os.path.join(dataset_path_flow, '**/*.npy'), recursive=True)
    pose_npy_files = glob.glob(os.path.join(dataset_path_pose, '**/*.npy'), recursive=True)
    # sort
    app_npy_files = sorted(app_npy_files, key=sort_files)
    flow_npy_files = sorted(flow_npy_files, key=sort_files)
    pose_npy_files = sorted(pose_npy_files, key=sort_files)

    test_app_npy_files = glob.glob(os.path.join(test_path_app, '**/*.npy'), recursive=True)
    test_flow_npy_files = glob.glob(os.path.join(test_path_flow, '**/*.npy'), recursive=True)
    test_pose_npy_files = glob.glob(os.path.join(test_path_pose, '**/*.npy'), recursive=True)

    test_app_npy_files = sorted(test_app_npy_files, key=sort_files)
    test_flow_npy_files = sorted(test_flow_npy_files, key=sort_files)
    test_pose_npy_files = sorted(test_pose_npy_files, key=sort_files)

    # dataset label
    test_label = np.load('[Your test label path]')
    test_label = test_label.flatten()


    # dataloader
    app_dataset = VideoFrameDataset(app_npy_files)
    app_dataloader = DataLoader(app_dataset, batch_size=64, shuffle=is_shuffle)
    flow_dataset = VideoFrameDataset(flow_npy_files)
    flow_dataloader = DataLoader(flow_dataset, batch_size=64, shuffle=is_shuffle)
    pose_dataset = VideoFrameDataset(pose_npy_files)
    pose_dataloader = DataLoader(pose_dataset, batch_size=64, shuffle=is_shuffle)

    test_app_dataset = test_VideoFrameDataset(full_frame_dir, test_path_app)
    test_app_dataloader = DataLoader(test_app_dataset, batch_size=1, shuffle=is_shuffle)
    test_flow_dataset = test_VideoFrameDataset(full_frame_dir, test_path_flow)
    test_flow_dataloader = DataLoader(test_flow_dataset, batch_size=1, shuffle=is_shuffle)
    test_pose_dataset = test_VideoFrameDataset(full_frame_dir, test_path_pose)
    test_pose_dataloader = DataLoader(test_pose_dataset, batch_size=1, shuffle=is_shuffle)


    # model_app = UNet(n_channels=12, half='fusion', n_classes=3).to(device)
    # model_pose = UNet(n_channels=12, half=is_pretrain, n_classes=3).to(device)
    # model_flow = UNet(n_channels=8, half=is_pretrain, n_classes=2).to(device)
    triple_unet = TripleUNet(n_channels=4, n_classes=1).to(device)
    # weight
    triple_unet.load_state_dict(torch.load('[Your weight path]'))

    num_epochs = 100
    max_auc_score = 0
    criterion = torch.nn.MSELoss()
    test_criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(triple_unet.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        triple_unet.train()
        running_loss = 0.0
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_loss4 = 0
        total_loss5 = 0
        total_loss6 = 0
        total_loss7 = 0
        total_loss8 = 0
        total_loss9 = 0

        for (inputs_app, labels_app), (inputs_flow, labels_flow), (inputs_pose, labels_pose) \
                in tqdm(zip(app_dataloader, flow_dataloader, pose_dataloader),
                    total=min(len(app_dataloader), len(flow_dataloader), len(pose_dataloader))):

            in_app = inputs_app.to(device, dtype=torch.float)
            l_app = labels_app.to(device, dtype=torch.float)
            in_flow = inputs_flow.to(device, dtype=torch.float)
            l_flow = labels_flow.to(device, dtype=torch.float)
            in_pose = inputs_pose.to(device, dtype=torch.float)
            l_pose = labels_pose.to(device, dtype=torch.float)
        
        
            logits_app, logits_flow_app, logits_pose_app, \
            flow_gathering_loss, flow_spreading_loss, flow_mem_reconstruction_loss, \
            pose_gathering_loss, pose_spreading_loss, pose_mem_reconstruction_loss = triple_unet(in_app, in_flow, in_pose)
        
            loss1 = criterion(logits_app, l_app)
            loss2 = criterion(logits_flow_app, l_flow)
            loss3 = criterion(logits_pose_app, l_pose)
            loss4 = flow_mem_reconstruction_loss
            loss5 = pose_mem_reconstruction_loss
            loss6 = flow_gathering_loss
            loss7 = pose_gathering_loss
            loss8 = flow_spreading_loss
            loss9 = pose_spreading_loss
        
            loss = loss1 + 0.01*loss2 + 0.1*loss3 + 0.01*loss4 + 0.01*loss5 + 0.1*loss6 + 0.1*loss7 + 0.01*loss8 + 0.01*loss9
        
            optimizer.zero_grad()
        
            loss.backward()
            optimizer.step()
        
            total_loss1 += loss1
            total_loss2 += loss2
            total_loss3 += loss3
            total_loss4 += loss4
            total_loss5 += loss5
            total_loss6 += loss6
            total_loss7 += loss7
            total_loss8 += loss8
            total_loss9 += loss9
        
            running_loss += loss.item()
            # print(f'\napp rec loss:{loss1}, flow rec loss:{loss2}, pose rec loss:{loss3}, flow mem loss:{loss4}, pose mem loss:{loss5}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(app_dataloader)}')
        print(f'\napp rec loss:{total_loss1 / len(app_dataloader)}, flow rec loss:{total_loss2 / len(app_dataloader)}, pose rec loss:{total_loss3 / len(app_dataloader)}, flow mem loss:{total_loss4 / len(app_dataloader)}, pose mem loss:{total_loss5 / len(app_dataloader)}, flow gather loss:{total_loss6 / len(app_dataloader)}, pose gather loss:{total_loss7 / len(app_dataloader)}, flow separate loss:{total_loss8 / len(app_dataloader)}, pose separate loss:{total_loss9 / len(app_dataloader)}')
        torch.save(triple_unet.state_dict(), f'model_state/{epoch}_{dataset}_model_app_weights_.pth')
        triple_unet.eval()

        min_frame_id = 0  
        max_frame_id = #[Set according to the number of frames of different datasets]
        losses_per_frame_1 = {frame_id: 0 for frame_id in range(min_frame_id, max_frame_id)}
        losses_per_frame_2 = {frame_id: 0 for frame_id in range(min_frame_id, max_frame_id)}
        losses_per_frame_3 = {frame_id: 0 for frame_id in range(min_frame_id, max_frame_id)}
        losses_per_frame_4 = {frame_id: 0 for frame_id in range(min_frame_id, max_frame_id)}
        losses_per_frame_5 = {frame_id: 0 for frame_id in range(min_frame_id, max_frame_id)}

        for (inputs_app, labels_app, frame_id), (inputs_flow, labels_flow, _), (
                inputs_pose, labels_pose, _) in tqdm(
            zip(test_app_dataloader, test_flow_dataloader, test_pose_dataloader),
            total=min(len(test_app_dataloader), len(test_flow_dataloader),
                      len(test_pose_dataloader))):
            in_app = inputs_app.to(device, dtype=torch.float)
            l_app = labels_app.to(device, dtype=torch.float)
            in_flow = inputs_flow.to(device, dtype=torch.float)
            l_flow = labels_flow.to(device, dtype=torch.float)
            in_pose = inputs_pose.to(device, dtype=torch.float)
            l_pose = labels_pose.to(device, dtype=torch.float)
            frame_id = frame_id.item()

            logits_app, logits_flow_app, logits_pose_app, \
            flow_gathering_loss, flow_spreading_loss, flow_mem_reconstruction_loss, \
            pose_gathering_loss, pose_spreading_loss, pose_mem_reconstruction_loss = triple_unet(in_app, in_flow, in_pose)

            image1 = logits_app.squeeze(0).cpu()
            image1 = image1.permute(1, 2, 0)
            image1 = image1.detach()
            image1 = image1.numpy()

            plt.imshow(image1)
            plt.axis('off')
            plt.show()

            image2 = l_app.squeeze(0).cpu()
            image2 = image2.permute(1, 2, 0)
            image2 = image2.detach()
            image2 = image2.numpy()

            plt.imshow(image2)
            plt.axis('off')
            plt.show()

            loss1 = criterion(logits_app, l_app)
            loss2 = criterion(logits_flow_app, l_flow)
            loss3 = criterion(logits_pose_app, l_pose)
            loss4 = flow_mem_reconstruction_loss
            loss5 = pose_mem_reconstruction_loss
            loss6 = flow_gathering_loss
            loss7 = pose_gathering_loss
            loss8 = flow_spreading_loss
            loss9 = pose_spreading_loss

            loss_5 = loss1 + 0.01*loss2 + 0.1*loss3 + 0.01*loss4 + 0.01*loss5

            loss_value_5 = loss_5.item()

            if loss_value_5 > losses_per_frame_5[frame_id]:
                losses_per_frame_5[frame_id] = loss_value_5


        max_frame_id = max(losses_per_frame_5.keys())

        losses_array = np.zeros(max_frame_id + 1)

        for frame_id, loss in losses_per_frame_5.items():
            losses_array[frame_id] = loss

        mean = np.mean(losses_array)
        std = np.std(losses_array)

        normalized_losses = (losses_array - mean) / std

        sigma = 3
        smoothed_losses = gaussian_filter1d(normalized_losses, sigma=sigma)

        y_true = test_label
        y_scores = smoothed_losses

        auc_score = roc_auc_score(y_true, y_scores)
        print(f"AUC Score: {auc_score}")
        torch.save(triple_unet.state_dict(), f'model_state/{epoch}_{dataset}_model_app_weights_{auc_score}.pth')
