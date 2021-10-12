import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataroot = "" #데이터터
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
opt.checkpoints_dir="" #pretrain model
opt.pairLst = ""    #입력쌍 csv
opt.L1_type = "l1_plus_perL1"
opt.n_layers_D = 3
opt.with_D_PP = 1
opt.with_D_PB = 1
otp.display_id = 0


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




