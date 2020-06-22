import torch
import matplotlib.pyplot as plt


def validate_model(model, testloader, device, num_images, classes):
    all_targets = torch.tensor([], dtype=torch.int64).to(device)
    all_preds = torch.tensor([], dtype=torch.int64).to(device)

    was_training = model.training
    model.eval()
    images_so_far = 0
    first_miss_class = 1
    first_correct_class = 1
    with torch.no_grad():
        for i, sample in enumerate(testloader['val']):
            # print(i)
            inputs = sample['image']
            labels = sample['target']
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            misclass_img = inputs[(labels != preds)]
            predicted_val = preds[(labels != preds)]
            actual_val = labels[labels != preds]
            # print(misclass_img.size())
            if first_miss_class == 1:
                misclass_data = misclass_img
                misclass_targets = actual_val
                misclass_preds = predicted_val
                first_miss_class = 0
            # elif first_correct_class == 1:
            #     correctclass_data = correctclass_img
            #     correctclass_targets = actual_val
            #     correctclass_preds = predicted_val
            #     first_correct_class = 0
            elif first_miss_class == 0:
                # one mis classified image already found now just append anymore
                misclass_data = torch.cat([misclass_data, misclass_img], dim=0)
                misclass_targets = torch.cat([misclass_targets, actual_val], dim=0)
                misclass_preds = torch.cat([misclass_preds, predicted_val], dim=0)
                # if len(misclass_data > 25):
                # break
            else:
                print("No Mis Classifications")
            all_targets = torch.cat((all_targets, labels), dim=0)
            all_preds = torch.cat((all_preds, preds), dim=0)
            cm, accuracy = create_confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 14))
    for j in range(inputs.size()[0]):
        images_so_far += 1
        ax = plt.subplot(num_images // 4, 4, images_so_far)
        ax.axis('off')
        ax.set_title('predicted:{} \n actual :{}'.format(classes[preds[j]], classes[labels[j]]))
        img = inputs.cpu().data[j]
        img = img.numpy().transpose(1, 2, 0)
        img = img / 2 + 0.5

        plt.imshow(img)

        if images_so_far == num_images:
            model.train(mode=was_training)
            return misclass_data, misclass_targets, misclass_preds, cm, accuracy
    model.train(mode=was_training)
    return misclass_data, misclass_targets, misclass_preds, cm, accuracy

def create_confusion_matrix(all_targets,all_preds):
  stacked = torch.stack((all_targets,all_preds),dim=1)
  # print(stacked.shape)
  cmt = torch.zeros(2,2,dtype=torch.int64)

  for p in stacked:
    tv,pv = p.tolist()
    cmt[tv,pv] = cmt[tv,pv]+1
  accuracy = (cmt[0][0].item()+cmt[1][1].item())/cmt.sum().item()
  return cmt, accuracy