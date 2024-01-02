import argparse
import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, file):
        self.filename = filename
        if not os.path.exists(filename):
            print("Error   : input file does not exist, exiting - {}".format(filename))
            exit(1)

        self.data = []
        with open(filename, "r") as f:
            for i,row in enumerate(f):
                row = row.strip().split(",")
                if i==0:
                    continue
                row = [float(x) for x in row]
                feature = row[:-1]
                label = row[-1]
                self.data.append([feature, label])

        # self.imgs_path = "Dog_Cat_Dataset/"
        # file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        # self.data = []
        # for class_path in file_list:
        #     class_name = class_path.split("/")[-1]
        #     for img_path in glob.glob(class_path + "/*.jpeg"):
        #         self.data.append([img_path, class_name])
        # print(self.data)
        # self.class_map = {"dogs" : 0, "cats": 1}
        # self.img_dim = (416, 416)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        feature_tensor = torch.from_numpy(np.array(feature))
        label_tensor = torch.tensor([label])
        return feature_tensor, label_tensor
        
        # img_path, class_name = self.data[idx]
        # img = cv2.imread(img_path)
        # img = cv2.resize(img, self.img_dim)
        # class_id = self.class_map[class_name]
        # img_tensor = torch.from_numpy(img)
        # img_tensor = img_tensor.permute(2, 0, 1)
        # class_id = torch.tensor([class_id])
        # return img_tensor, class_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="sample.csv")
    args = parser.parse_args()
    filename = args.filename

    print("Filename: {}".format(filename))

    dataset = CustomDataset(filename)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for features, labels in data_loader:
        print("Batch of features has shape: ",features.shape)
        print("Batch of labels has shape: ", labels.shape)
