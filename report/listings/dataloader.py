class dataset_statistic(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.files[index])
        image = Image.open(img_path)
        label = labels[img_path.split('/')[-1].split('_')[0]]

        if self.transforms:
            for t in self.transforms:
                image = t(image)
        return (image, torch.tensor(label))
    
val_data = dataset_statistic(path_test, transforms=base_transforms)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)