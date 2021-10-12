import time
import os
from options.test_options import TestOptions
import easydict
#from data.data_loader import CreateDataLoader
#from models.models import create_model
#from util.visualizer import Visualizer
#from util import html
#import time

opt = easydict.EasyDict({
 
        "BP_input_nc": 18,
        "D_n_downsampling": 2,
        "G_n_downsampling": 2,
        "P_input_nc": 3,
        "aspect_ratio": 1.0,
        "batchSize": 1,
        "checkpoints_dir": "./BiGraphGAN/scripts/checkpoints",
        "dataroot": "./SelectionGAN/person_transfer/datasets/market_data/",
        "dataset_mode": "keypoint",
        "display_id": 0,
        "display_port": 8097,
        "display_winsize": 256,
        "fineSize": 256,
        "gpu_ids": [0],
        "how_many": 200,
        "init_type": "normal",
        "input_nc": 3,
        "isTrain": False,
        "loadSize": 286,
        "max_dataset_size": float("inf"),
        "model": "BiGraphGAN",
        "nThreads": 2,
        "n_layers_D": 3,
        "name": "market_pretrained",
        "ndf": 64,
        "ngf": 64,
        "no_dropout": False,
        "no_flip": True,
        "norm": "batch",
        "ntest": float("inf"),
        "output_nc": 3,
        "padding_type": "reflect",
        "pairLst": "./SelectionGAN/person_transfer/datasets/market_data/market-pairs-test.csv",
        "phase": "test",
        "resize_or_crop": "no",
        "results_dir": "./results/",
        "serial_batches": False,
        "use_flip": 0,
        "which_direction": "AtoB",
        "which_epoch": "latest",
        "which_model_netD": "resnet",
        "which_model_netG": "Graph",
        "with_D_PB": 1,
        "with_D_PP": 1
 
})

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataroot = "/content/dataset" #데이터터
opt.name = "market_exp"
opt.model = "BiGraphGAN"
opt.lambda_GAN = 5
opt.lambda_A = 10
opt.lambda_B = 10
opt.dataset_mode = "keypoint"
opt.n_layers = 3
opt.norm = "batch"
opt.batchSize = 32
opt.resize_or_crop = "no"
opt.gpu_ids = 0
opt.BP_input_nc = 18

opt.no_flip
opt.no_lsgan

opt.which_model_netG = "Graph"
opt.niter = 500
opt.niter_decay = 200
opt.checkpoints_dir="/content/drive/MyDrive/datasets/bigraph" #pretrain model
opt.pairLst = "/content/pairs_data.csv"    #입력쌍 csv
opt.L1_type = "l1_plus_perL1"
opt.n_layers_D = 3
opt.with_D_PP = 1
opt.with_D_PB = 1
otp.display_id = 0

print(opt)
'''
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print(opt.how_many)
print(len(dataset))

model = model.eval()
print(model.training)

opt.how_many = 999999
# test
for i, data in enumerate(dataset):
    print(' process %d/%d img ..'%(i,opt.how_many))
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    print(endTime-startTime)
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    img_path = [img_path]
    print(img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()




'''