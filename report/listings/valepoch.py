def test(model, criterion, val_loader):
    model.eval()

    accuracy_log, loss_log = [], []
    accuracy_log_wb, loss_log_wb = 0, 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, labels)

        labels_pred = torch.max(logits, dim=1)[1].cpu().detach().numpy()
        labels_true = labels.cpu().detach().numpy()

        accuracy_log.append(np.mean(labels_pred == labels_true))
        loss_log.append(loss.item())
        loss_log_wb += loss.item() * images.shape[0]
        accuracy_log_wb += (logits.argmax(dim=1) == labels).sum().item()

    accuracy_log_wb /= len(val_loader.dataset)
    loss_log_wb /= len(val_loader.dataset)

    return accuracy_log, loss_log, accuracy_log_wb, loss_log_wb,