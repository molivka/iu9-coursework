def show_image(data):
    img = data[0]
    label = data[1].item()
    print(label)
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img.permute(1, 2, 0))