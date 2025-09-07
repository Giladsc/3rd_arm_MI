import torch
from torchvision import datasets, models, transforms
from torchvision.models import AlexNet_Weights
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models import ConvNeXt_Small_Weights
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.models import EfficientNet_V2_S_Weights
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from functools import partial


def alexnet_train(path='Scalograms/Denoised/', optimizer='adamw', learning_rate=0.001, bs=16, save_model=False):
    '''
    Function to set up the AlexNet model with specified hyperparameters for transfer learning and train it on the provided dataset.
    
    Parameters:
        path (str): Path to the dataset directory.
        optimizer (str): Optimizer to use ('adam', 'adamw', 'rmsprop', 'sgdm').
        learning_rate (float): Learning rate for the optimizer.
        bs (int): Batch size for training.
        save_model (bool): Whether to save the best model.
    Returns:
        best_valid_acc (float): The best validation accuracy achieved during training. 
                                It is also the validation accuracy of the saved model if `save_model` is True. 
    '''
    
    # Applying Transforms to the Data
    image_transforms = { 
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Load the Data
    # Set train and valid directory paths

    train_directory = os.path.join(path, 'Train')
    valid_directory = os.path.join(path, 'Valid')

    # Number of classes
    num_classes = len(os.listdir(valid_directory))
    # print(num_classes)

    # Load Data from folders
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }

    # Get a mapping of the indices to the class names.
    # idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
    # print(idx_to_class)

    # Size of Data, to be used for calculating mean validation accuracy
    valid_data_size = len(data['valid'])

    # Create iterators for the Data loaded using DataLoader module
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)

    alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)

    # Freeze model parameters
    for param in alexnet.parameters():
        param.requires_grad = False
    
    # Change the final layer of Alexnet Model for Transfer Learning
    alexnet.classifier[4] = nn.Linear(4096, 8192)
    alexnet.classifier[6] = nn.Linear(8192, num_classes) # (8192, num_classes) if previous line is uncommented, otherwise (4096, num_classes)
    ### alexnet.classifier.add_module("7", nn.LogSoftmax(dim = 1))
    
    # Define Optimizer and Loss Function
    ### loss_func = nn.NLLLoss()
    loss_func = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for multi-class classification
    if optimizer == 'adam':
        optimizer = optim.Adam(alexnet.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(alexnet.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(alexnet.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'sgdm':
        optimizer = optim.SGD(alexnet.classifier.parameters(), lr=learning_rate, momentum=0.9)
    
    # Train the model while monitoring the validation accuracy
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alexnet = alexnet.to(device)
    best_valid_acc = 0.0
    for epoch in range(30):
        alexnet.train()
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = alexnet(inputs)
            loss_func(outputs, labels).backward()
            optimizer.step()

        with torch.no_grad():
            alexnet.eval()
            for i, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = alexnet(inputs)
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                valid_acc += correct_counts.sum().item()

        mean_valid_acc = valid_acc / valid_data_size
        
        if mean_valid_acc > best_valid_acc:
            best_valid_acc = mean_valid_acc
            if save_model:
                torch.save(alexnet.state_dict(), 'final_model.pt')
    return best_valid_acc



def shufflenet_v2_train (path='Scalograms/Denoised/', optimizer='adamw', learning_rate=0.001, bs=16, save_model=False):
    '''
    Function to set up the ShuffleNetV2 model with specified hyperparameters for transfer learning and train it on the provided dataset.
    
    Parameters:
        path (str): Path to the dataset directory.
        optimizer (str): Optimizer to use ('adam', 'adamw', 'rmsprop', 'sgdm').
        learning_rate (float): Learning rate for the optimizer.
        bs (int): Batch size for training.
        save_model (bool): Whether to save the best model.
    Returns:
        best_valid_acc (float): The best validation accuracy achieved during training. 
                                It is also the validation accuracy of the saved model if `save_model` is True. 
    '''
    # Applying Transforms to the Data
    image_transforms = { 
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Load the Data
    train_directory = os.path.join(path, 'Train')
    valid_directory = os.path.join(path, 'Valid')
    num_classes = len(os.listdir(valid_directory))

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }
    valid_data_size = len(data['valid'])
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)

    shufflenet = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)

    for param in shufflenet.parameters():
        param.requires_grad = False
    
    shufflenet.fc = nn.Linear(shufflenet.fc.in_features, num_classes)
    # shufflenet.add_module("logsoftmax", nn.LogSoftmax(dim=1))
    # loss_func = nn.NLLLoss()
    loss_func = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for multi-class classification

    if optimizer == 'adam':
        optimizer = optim.Adam(shufflenet.fc.parameters(), lr=learning_rate)
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(shufflenet.fc.parameters(), lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(shufflenet.fc.parameters(), lr=learning_rate)
    elif optimizer == 'sgdm':
        optimizer = optim.SGD(shufflenet.fc.parameters(), lr=learning_rate, momentum=0.9)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shufflenet = shufflenet.to(device)
    best_valid_acc = 0.0
    for epoch in range(30):
        shufflenet.train()
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = shufflenet(inputs)
            loss_func(outputs, labels).backward()
            optimizer.step()

        with torch.no_grad():
            shufflenet.eval()
            for i, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = shufflenet(inputs)
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                valid_acc += correct_counts.sum().item()

        mean_valid_acc = valid_acc / valid_data_size
        
        if mean_valid_acc > best_valid_acc:
            best_valid_acc = mean_valid_acc
            if save_model:
                torch.save(shufflenet.state_dict(), 'final_model.pt')
    return best_valid_acc


def convnext_train(path='Scalograms/Denoised/', optimizer='adam', learning_rate=0.001, bs=16, model_size='t', save_model=False):
    '''
    Function to set up the ConvNeXt-Large model with specified hyperparameters for transfer learning and train it on the provided dataset.

    Parameters:
        path (str): Path to the dataset directory.
        optimizer (str): Optimizer to use ('adam', 'adamw', 'rmsprop', 'sgdm').
        learning_rate (float): Learning rate for the optimizer.
        bs (int): Batch size for training.
        model_size (str): Size of the ConvNeXt model ('t' for tiny, 's' for small).
        save_model (bool): Whether to save the best model.
    Returns:
        best_valid_acc (float): The best validation accuracy achieved during training.
    '''

    # Applying Transforms to the Data
    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Load the Data
    train_directory = os.path.join(path, 'Train')
    valid_directory = os.path.join(path, 'Valid')
    num_classes = len(os.listdir(valid_directory))

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }
    valid_data_size = len(data['valid'])
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)
    
    if model_size == 't':
        convnext = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    elif model_size == 's':
        convnext = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)

    # Freeze model parameters
    for param in convnext.parameters():
        param.requires_grad = False

    # Replace the classifier head for transfer learning
    convnext.classifier[2] = nn.Linear(convnext.classifier[2].in_features, num_classes)

    loss_func = nn.CrossEntropyLoss()

    if optimizer == 'adam':
        optimizer = optim.Adam(convnext.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(convnext.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(convnext.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'sgdm':
        optimizer = optim.SGD(convnext.classifier.parameters(), lr=learning_rate, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    convnext = convnext.to(device)
    best_valid_acc = 0.0
    for epoch in range(30):
        convnext.train()
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = convnext(inputs)
            loss_func(outputs, labels).backward()
            optimizer.step()

        with torch.no_grad():
            convnext.eval()
            for i, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = convnext(inputs)
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                valid_acc += correct_counts.sum().item()

        mean_valid_acc = valid_acc / valid_data_size

        if mean_valid_acc > best_valid_acc:
            best_valid_acc = mean_valid_acc
            if save_model:
                torch.save(convnext.state_dict(), 'final_model.pt')
    return best_valid_acc

def efficientnetv2_train(path='Scalograms/Denoised/', optimizer='adamw', learning_rate=0.001, bs=16, save_model=False):
    '''
    Function to set up the EfficientNet_V2_S model with specified hyperparameters for transfer learning and train it on the provided dataset.

    Parameters:
        path (str): Path to the dataset directory.
        optimizer (str): Optimizer to use ('adam', 'adamw', 'rmsprop', 'sgdm').
        learning_rate (float): Learning rate for the optimizer.
        bs (int): Batch size for training.
        save_model (bool): Whether to save the best model.
    Returns:
        best_valid_acc (float): The best validation accuracy achieved during training.
    '''

    # Custom transforms: resize, center crop, rescale, normalize
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize(384, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(384, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the Data
    train_directory = os.path.join(path, 'Train')
    valid_directory = os.path.join(path, 'Valid')
    num_classes = len(os.listdir(valid_directory))

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }
    valid_data_size = len(data['valid'])
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)

    # Load EfficientNet_V2_S with pretrained weights
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head for transfer learning
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    loss_func = nn.CrossEntropyLoss()

    if optimizer == 'adam':
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.classifier.parameters(), lr=learning_rate)
    elif optimizer == 'sgdm':
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_valid_acc = 0.0
    for epoch in range(30):
        model.train()
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_func(outputs, labels).backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for i, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                valid_acc += correct_counts.sum().item()

        mean_valid_acc = valid_acc / valid_data_size

        if mean_valid_acc > best_valid_acc:
            best_valid_acc = mean_valid_acc
            if save_model:
                torch.save(model.state_dict(), 'final_model.pt')
    return best_valid_acc

def train_model(model_name, path='Scalograms/Denoised/', optimizer='adamw', learning_rate=0.001, bs=16, save_model=False):
    '''
    Function to train a specified model with given hyperparameters on the provided dataset.

    Parameters:
        model_name (str): Name of the model to train ('alexnet', 'shufflenet_v2', 'convnext_t', 'convnext_s', 'efficientnet_v2').
        path (str): Path to the dataset directory.
        optimizer (str): Optimizer to use ('adam', 'adamw', 'rmsprop', 'sgdm').
        learning_rate (float): Learning rate for the optimizer.
        bs (int): Batch size for training.
        save_model (bool): Whether to save the best model.
    
    Returns:
        best_valid_acc (float): The best validation accuracy achieved during training.
    '''
    if model_name == 'alexnet':
        return alexnet_train(path, optimizer, learning_rate, bs, save_model)
    elif model_name == 'shufflenet_v2':
        return shufflenet_v2_train(path, optimizer, learning_rate, bs, save_model)
    elif model_name == 'convnext_t':
        return convnext_train(path, optimizer, learning_rate, bs, model_size='t', save_model=save_model)
    elif model_name == 'convnext_s':
        return convnext_train(path, optimizer, learning_rate, bs, model_size='s', save_model=save_model)
    elif model_name == 'efficientnet_v2':
        return efficientnetv2_train(path, optimizer, learning_rate, bs, save_model)
    return None

def evaluate_final_model(model_name, test_path='Scalograms/Denoised/Test', model_path='final_model.pt', num_classes=2):
    '''
    Loads final model from specified directory and evaluates it on the test dataset.

    Parameters:
        model_name (str): Name of the model to evaluate ('alexnet' / 'shufflenet' / 'convnext' / 'efficientnet_v2').
        path (str): Path to the dataset directory.
        model_path (str): Path to the saved model.
    Returns:
        test_acc (float): The accuracy of the model on the test dataset.
    '''
    
    # Load the saved model
    if model_name == 'alexnet':
        model = models.alexnet()
        model.classifier[4] = nn.Linear(4096, 8192)
        model.classifier[6] = nn.Linear(8192, 2)
        ### model.classifier.add_module("7", nn.LogSoftmax(dim=1))
    elif model_name == 'shufflenet_v2':
        model = models.shufflenet_v2_x2_0()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        ### model.add_module("logsoftmax", nn.LogSoftmax(dim=1))
    elif model_name == 'convnext_t':
        model = models.convnext_tiny()
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == 'convnext_s':
        model = models.convnext_small()
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == 'efficientnet_v2':
        model = models.efficientnet_v2_s()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    
    # Load the test data
    if model_name == 'efficientnet_v2':
        test_data = datasets.ImageFolder(root=test_path, transform=transforms.Compose([
            transforms.Resize(384, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    else:
        test_data = datasets.ImageFolder(root=test_path, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_acc = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            test_acc += correct_counts.sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    test_acc /= len(test_data)
    conf_mat = confusion_matrix(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions, target_names=test_data.classes)
    return test_acc, conf_mat, class_report