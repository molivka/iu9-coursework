from google.colab import drive
drive.mount('/content/drive', force_remount=True)
# загрузка .zip файла с изображениями
!unzip '/content/drive/MyDrive/graphs.zip' -d '/content/drive/MyDrive/graphs'
# создание папок для распрделения данных
!mkdir '/content/drive/MyDrive/graphs_classes'
!mkdir '/content/drive/MyDrive/graphs_classes/train'
!mkdir '/content/drive/MyDrive/graphs_classes/test'

path_train = '/content/drive/MyDrive/graphs_classes/train'
path_test = '/content/drive/MyDrive/graphs_classes/test'

# метки для изображений
labels = {
    'binom': 0,
    'geom': 1,
    'poisson': 2,
    'norm': 3,
    'pareto': 4,
    'vigner': 5
}

images = []
for image in os.listdir('/content/drive/MyDrive/graphs'):
    print(image.split('_')[0])
    if image.split('_')[0] in labels:
        images.append(image)

shuffle(images)

for i in range(1, len(images)):
    image = images[i]
    if i < len(images) * 0.7:
        os.replace(f'/content/drive/MyDrive/graphs/{image}', f'{path_train}/{image}')
        shutil.move(f'/content/drive/MyDrive/graphs/graphs/{image}', f'{path_train}/{image}')
    else:
        os.replace(f'/content/drive/MyDrive/graphs/{image}', f'{path_test}/{image}')
        shutil.move(f'/content/drive/MyDrive/graphs/graphs/{image}', f'{path_test}/{image}')