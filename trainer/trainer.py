import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class Trainer:
    def __init__(self, model, device, loss) -> None:
        self.model = model
        self.device = device
        self.loss = loss


    def train(self, train_loader, lr, epochs, test_loader):
        # store the losses and accs for plotting
        losses, train_accs, test_accs = [], [], []

        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.5e-6)
        if self.loss == 'MSE':
            criterion = MSELoss()
        elif self.loss == 'CrossEntropy':
            criterion = CrossEntropyLoss()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.model.train()

            loss_epoch = 0.0
            n_batches = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.to(torch.float)
                # target = target.to(torch.float)
                optimizer.zero_grad()

                # one-hot encoding targets to match the shape requirement of Softmax.
                if self.loss == 'MSE':
                    target = F.one_hot(target, num_classes=10)
                    target = target.to(torch.float)
                
                outputs = criterion(self.model(data), target)
                outputs.backward()
                optimizer.step()
                
                loss_epoch += outputs.item()
                n_batches += 1

            # test
            train_accs.append(float(self.test(train_loader)))
            test_accs.append(float(self.test(test_loader)))
            losses.append(loss_epoch/n_batches)

            epoch_train_time = time.time() - epoch_start_time            
            print('Epoch {}/{}\tTime: {:.3f}\tLoss: {:.6f}\tTrain acc: {:6f}\tTest acc: {:6f}'
                .format(epoch+1, epochs, epoch_train_time, loss_epoch/n_batches, train_accs[-1], test_accs[-1]))
        train_time = time.time() - start_time
        print('Training time: %.3f' % train_time)
        print('Finished training.')
        return losses, train_accs, test_accs

    def test(self, test_loader):
        self.model.eval()
        correct = torch.zeros(1).squeeze().to(self.device)
        total = torch.zeros(1).squeeze().to(self.device)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.to(torch.float)
                outputs = self.model(data)

                prediction = torch.argmax(outputs, 1)
                correct += (prediction == target).sum().float()
                total += len(target)


            acc = (correct/total).cpu().detach().data.numpy()
            # print('Test Acc: %.6f' % acc)
        return acc

    def examine_features(self, test_loader):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.to(torch.float)

                self.model.conv3.register_forward_hook(get_activation('conv3'))
                outputs = self.model(data)
                break
            features = activation['conv3'].cpu().detach().data.numpy()
            target = target.cpu().detach().data.numpy()
        return features[:10], target[:10]

            

        