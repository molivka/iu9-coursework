# перспективное преобразование, случайный перспективный сдвиг на 100*s%
s = torch.rand(1)
perspective_transform = [T.Resize(for_resize), T.RandomPerspective(distortion_scale=s), T.ToTensor()]
# поворот изображения на t градусов
t = torch.randint(0, 360, (1,)).item()
rotation_transform = [T.Resize(for_resize), T.RandomRotation(t), T.ToTensor()]
# изменение свойств цвета изображения
br = 1 # яркость
con = torch.randint(0, 100, (1,)).item() # контраст
sat = torch.randint(0, 100, (1,)).item() # насыщенность
hue = min(0.5, torch.rand(1).item()) # оттенок
color_transform = [T.Resize(for_resize), T.ColorJitter(brightness=br, contrast=con, saturation=sat, hue=hue), T.ToTensor()]
# базовое преобразование
base_transforms = [T.Resize(for_resize), T.ToTensor()]


train_data = dataset_statistic(path_train, transforms=base_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)