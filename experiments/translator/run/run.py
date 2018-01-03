import os
import time

from torch.autograd import Variable


def run(n_epoch,
        batch_size,
        n_layers,
        h,
        dropout,
        mask,
        d_pw_ffn,
        epsilon,
        logdir):
    """"""
    model = Transformer()
    if torch.is_available():
        model.cuda()
    criterion
    optimizer
    scheduler

    logger = JSONLogger(filedir=logdir,
                        filename="results",
                        config={"n_epoch": n_epochs,
                                "batch_size" batch_size,
                                "n_layers": n_layers,
                                "h": h,
                                "dropout": dropout,
                                "mask": mask,
                                "d_pw_ffn": d_pw_ffn,
                                "epsilon": epsilon,
                                "logdir": logdir})

    best_loss = float("inf")
    for epoch in range(n_epoch):
        scheduler.step()
        since = time.time()
        train_loss _train(model=model,
                          train_data=,
                          criterion=loss)
        eval_loss = _evaluate(model=model,
                              eval_data=,
                              criterion=loss)

        event = {"epoch": epoch,
                 "train_loss": train_loss,
                 "eval_loss": eval_loss,
                 "time_spent": time.time() - since}

        print(",".join(["epoch: {epoch}",
                        "train loss: {train_loss}",
                        "eval loss: {eval_loss}",
                        "time spent: {time_spent}"]).format(**event))

        if eval_loss < best_loss:
            best_loss = eval_loss
            path = os.path.join(filepath, filename) + ".pth.tar"
            with open(path, "w+") as fp:
                torch.save(model.state_dict(), fp)
            logger.add_event(event=event)

    logger.save_log()


def _train(model, train_data, criterion, optimizer):
    """"""
    model.train()
    running_loss = 0.0
    obs_count = 0
    for x, y in train_data:
        x_var, y_var = Variable(x), Variable(y)

        # Forward pass
        y_pred = model()
        loss = criterion()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs_count += x_var.size(1)
        running_loss += loss.data[0]

    return running_loss / obs_count


def _evaluate(model, eval_data, criterion):
    """"""
    model.eval()
    running_loss = 0.0
    obs_count = 0
    for x, y in eval_data:
        x_var, y_var = Variable(x, volatile=True), Variable(y, volatile=True)

        # Forward pass
        y_pred = model()
        loss = criterion()

        obs_count += x_var.size(1)
        running_loss += loss.data[0]

    return running_loss / obs_count
