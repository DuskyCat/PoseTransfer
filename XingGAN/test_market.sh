export CUDA_VISIBLE_DEVICES=0; 
python test.py --dataroot ./SelectionGAN/person_transfer/datasets/market_data/ --name market_XingGAN --model XingGAN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG Xing --checkpoints_dir ./checkpoints --pairLst ./SelectionGAN/person_transfer/datasets/market_data/market-pairs-test.csv --which_epoch latest --results_dir ./results --display_id 0