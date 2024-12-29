def train(model, criterion, train_loader, val_loader, optimizer, num_epoches, batch_size, name, project_name):
    train_accuracy_log, train_loss_log = [], []
    val_accuracy_log, val_loss_log = [], []

    for epoch in range(1, num_epoches + 1):
        train_accuracy, train_loss, train_accuracy_wb, train_loss_wb = train_epoch(model, criterion, train_loader, optimizer, name)
        val_accuracy, val_loss, val_accuracy_wb, val_loss_wb = test(model, criterion, val_loader)

        train_accuracy_log.extend(train_accuracy)
        train_loss_log.extend(train_loss)
        steps = train_loader.dataset.__len__() / batch_size

        val_accuracy_log.append((steps * epoch, np.mean(val_accuracy)))
        val_loss_log.append((steps * epoch, np.mean(val_loss)))

        clear_output()
        plot_logs(train_accuracy_log, val_accuracy_log, train_loss_log, val_loss_log)

    info = {
        'train_acc': np.mean(train_accuracy_log),
        'test_acc': val_accuracy_log[-1],
        'train_loss': np.mean(val_loss_log),
        'test_loss': val_loss_log[-1]
    }

    metrics = {
            "train accuracy": train_accuracy_wb / len(train_loader.dataset),
            "train loss": train_loss_wb / len(train_loader.dataset),
            "val accuracy": val_accuracy_wb,
            "val loss": val_loss_wb
    }
    wandb.log(metrics)

    return info