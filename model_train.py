import torch
from tqdm import tqdm
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

#LEARNING_RATE = 1e-05
loss_function = torch.nn.CrossEntropyLoss()

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item() # item() returns value of a tensor
    return n_correct


def train(model,optimizer,epochs,training_loader):

    no_increase = []
    peak = 0
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(epochs):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        loss_list = []
        accuracy_list = []
        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device)
            target_big_val, target_big_idx = torch.max(targets, dim=1)
            outputs = model(ids, mask, token_type_ids)  # pooler outputs of each sequence
            loss = loss_function(outputs, target_big_idx)
            tr_loss += loss.item()
            pred_big_val, pred_big_idx = torch.max(outputs.data, dim=1)  # big_idx is the location of max val found
            n_correct += calcuate_accuracy(pred_big_idx, target_big_idx)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # print out loss every 100 steps
            if _ % 100 == 0 and _ != 0:
                loss_step = round(tr_loss / nb_tr_steps, 4)
                accu_step = round(n_correct / nb_tr_examples, 4)
                print(f"Training Loss per {_} steps: {loss_step}")
                print(f"Training Accuracy per {_} steps: {accu_step}")
                loss_list.append(loss_step)
                accuracy_list.append(accu_step)
                #valid_accu = validation(model, valid_loader)
                if accu_step > peak:
                    peak = accu_step
                    no_increase = []
                else:
                    no_increase.append(0)
                if len(no_increase) == 20:
                    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
                    #epoch_loss = tr_loss / nb_tr_steps
                    #epoch_accu = n_correct / nb_tr_examples
                    print('early stopping')
                    print(f"Training Loss after {epoch} incomplete epoch: {loss_step}")
                    print(f"Training Accuracy after {epoch} incomplete epoch: {accu_step}")
                    #loss_file = open("sentence_roberta_training_loss_es" , 'wb')
                    #accu_file = open( "sentence_roberta_training_accuracy_es" ,"wb")
                    #pickle.dump(loss_list, loss_file)
                    #pickle.dump(accuracy_list, accu_file)
                    #loss_file.close()
                    #accu_file.close()
                    return model, loss_step, optimizer


            optimizer.zero_grad()  # clear out old gradients that already have been used to update weights
            loss.backward()  # calculate the gradient of loss
            # # When using GPU
            optimizer.step()  # update the parameters
        #print out accuracies and loss after each epoch

        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
        epoch_loss = round(tr_loss / nb_tr_steps,4)
        epoch_accu = round(n_correct/ nb_tr_examples,4)
        print(f"Training Loss after {epoch } epochs: {epoch_loss}")
        print(f"Training Accuracy after {epoch } epochs: {epoch_accu}")

    return model, epoch_loss, optimizer
