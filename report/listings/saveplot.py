def save_plot(x, y, name, ind, is_fill=0, xmin=0, xmax=0, ylim=1):
    plt.ylim((None, ylim))
    if is_fill:
        plt.xlim((xmin, xmax))
        plt.plot(x, y)
        plt.fill_between(x, y)
    else:
        plt.bar(x, y)
    plt.savefig(f"{path}/{name}_{ind}.jpg")
    plt.show()
    plt.close()