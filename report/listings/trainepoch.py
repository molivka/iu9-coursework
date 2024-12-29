def train_epoch(model, criterion, train_loader, optimizer, name):
    model.train()

    accuracy_log, loss_log = [], []
    accuracy_log_wb, loss_log_wb = 0, 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()


        labels_pred = torch.max(logits, dim=1)[1].cpu().detach().numpy()
        labels_true = labels.cpu().detach().numpy()


        accuracy_log.append(np.mean(labels_pred == labels_true))
        loss_log.append(loss.item())
        accuracy_log_wb += (logits.argmax(dim=1) == labels).sum().item()
        loss_log_wb += loss.item() * images.shape[0]

        metrics = {
            f"{name} batch-train loss": loss.item()
        }
        wandb.log(metrics)

    accuracy_log_wb /= len(train_loader.dataset)
    loss_log_wb /= len(train_loader.dataset)

    return accuracy_log, loss_log, accuracy_log_wb, loss_log_wb