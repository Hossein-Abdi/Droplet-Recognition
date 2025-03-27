import pandas as pd
import torch
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Dinov2Model 
import torch.optim as optim
from torchvision import datasets
import torch.nn as nn
import glob
import os
from tqdm import tqdm




#### Hyperparameters ##########################
num_epochs = 100
batch_size = 10
lr = 0.001
max_circles = 30 
####################################################




# Device 
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# device = torch.device("cpu")
print("device =", device)


dtype = torch.float32




class TiffDataset(Dataset):
    def __init__(self, folder_path, transform=None, max_circles=30):
        self.image_paths = glob.glob(os.path.join(folder_path, "*.tif"))
        self.transform = transform
        self.max_circles = max_circles

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Use "RGB" for colorful image
        if self.transform:
            image = self.transform(image)
        return image, os.path.splitext(os.path.basename(img_path))[0]
    







class DINOv2CircleDetectionModel(nn.Module):
    def __init__(self, max_circles=30):
        super(DINOv2CircleDetectionModel, self).__init__()
        self.max_circles = max_circles

        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")

        self.fc = nn.Linear(self.dinov2.config.hidden_size, max_circles * 3)  # Output max_circles * 3 values

    def forward(self, x):
        outputs = self.dinov2(x)  # Shape: (batch_size, sequence_length, hidden_size)

        cls_token = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

        output = self.fc(cls_token)  # Shape: (batch_size, max_circles * 3)

        # Reshape to (batch_size, max_circles, 3)
        output = output.view(x.size(0), self.max_circles, 3)
        return output









model = DINOv2CircleDetectionModel(max_circles=max_circles).to(device).to(dtype)
criterion = nn.MSELoss(reduction='none')  # Use reduction='none' to compute loss per circle
optimizer = optim.Adam(model.parameters(), lr=lr)







# Load the Excel file
root = os.path.dirname(os.path.abspath(__file__)) 
file_excel = pd.read_excel(root+"/myData.xlsx") 






#Image:
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])
dataset = TiffDataset(folder_path=root + "/raw_image/", transform=transform, max_circles=30)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)






def main():
    model.train()
    for epoch in range(num_epochs):
        for batch_images, batch_filenames in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_images = batch_images.to(device)

            batch_circles = torch.zeros((batch_size, max_circles, 3)).to(device).to(dtype)
            batch_masks = torch.zeros((batch_size, max_circles)).to(device).to(dtype)
            

            for i in range(batch_size):
                mask = torch.zeros((max_circles,)).to(dtype)
                data_np = file_excel.to_numpy()
                data_tensor = torch.tensor(data_np).to(dtype)
                
                mask_excel = (data_tensor[:, 0:1] == torch.tensor(float(batch_filenames[i]))).squeeze() 
                relevant_data = data_tensor[mask_excel]
                num_circles = min(len(relevant_data), max_circles)
                mask[:num_circles] = 1.0
                batch_masks[i] = mask
                
                for j in range(num_circles):
                    batch_circles[i, j, 0] = relevant_data[j, 5]  # x
                    batch_circles[i, j, 1] = relevant_data[j, 6]  # y
                    batch_circles[i, j, 2] = relevant_data[j, 1] / 2  # radius

                

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_circles) * batch_masks.unsqueeze(-1)
            loss = loss.sum() / batch_masks.sum()
            loss.backward()
            optimizer.step()
            print("Loss = ", loss.item())






if __name__ == '__main__':
    main()


    for image, filename in tqdm(test_loader, desc="Testing"):
        model.eval()
        image = image.to(device).to(dtype)
        print("image name = ", filename)
        output = model(image)
        
        x = output.squeeze(0)[:, 0].detach().numpy()
        y = output.squeeze(0)[:, 1].detach().numpy()
        r = output.squeeze(0)[:, 2].detach().numpy()


        plt.imshow(image.squeeze(0)[0], cmap="gray") 
        plt.axis("off")

        for i in range(7):
            circle = plt.Circle((x[i], y[i]), r[i], color='r', fill=False, linewidth=2)
            plt.gca().add_patch(circle)


        plt.show()

