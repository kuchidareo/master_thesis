from torchvision.models import mobilenet_v3_small
from ultralytics.nn.tasks import DetectionModel

from loaders import casas as casas_loader, aep as aep_loader, visdrone as visdrone_loader, wisdm as wisdm_loader, german_traffic as german_traffic_loader
from models import casas as casas_model, aep as aep_model, ecg as ecg_model, mnist as mnist_model, wisdm as wisdm_model, uci_har as uci_har_model, german_traffic as germann_traffic_model, trashnet as trashnet_model

def get_dataset_info(dataset_name):
    match dataset_name:
        case "mnist":
            model = mnist_model.MnistNet()
            num_classes = 10
            category = "huggingface"
            dataset = None
        case "cifar10":
            model = mobilenet_v3_small(num_classes=10)
            num_classes = 10
            category = "huggingface"
            dataset = None
        case "trashnet":
            model = trashnet_model.SimpleCNN()
            num_classes = 6
            category = "huggingface"
            dataset = None
        case "german_traffic":
            # model = mobilenet_v3_small(num_classes=10)
            # num_classes = 43
            # category = "huggingface_dataset"
            pass
        case "ecg":
            model = ecg_model.EcgConv1d()
            num_classes = 10
            category = "custom"
            dataset = None
        case "uci_har":
            model = uci_har_model.HARNet()
            num_classes = 10
            category = "custom"
            dataset = None
        case "casas":
            model = casas_model.BiLSTMModel()
            num_classes = 12
            category = "fedaiot"
            dataset = casas_loader.load_dataset()
        case "aep":
            model = aep_model.MLP()
            num_classes = 10
            category = "fedaiot"
            dataset = aep_loader.load_dataset()
        case "visdrone": 
            model = DetectionModel(cfg="yolov8.yaml")
            num_classes = 12
            category = "fedaiot"
            dataset = visdrone_loader.load_dataset()
        case "wisdm_watch":
            model = wisdm_model.LSTM_NET()
            num_classes = 12
            category = "fedaiot"
            dataset = wisdm_loader.load_dataset(reprocess=False, modality='watch')
        case "wisdm_phone":
            model = wisdm_model.LSTM_NET()
            num_classes = 12
            category = "fedaiot"
            dataset = wisdm_loader.load_dataset(reprocess=False, modality='phone')
        
    return model, num_classes, category, dataset

