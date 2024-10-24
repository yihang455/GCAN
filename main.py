import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver


def main(args):
    cudnn.benchmark = True

    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)
    #     print('Create path : {}'.format(args.save_path))

    # if args.result_fig:
    #     fig_path = os.path.join(args.save_path, 'fig')
    #     if not os.path.exists(fig_path):
    #         os.makedirs(fig_path)
    #         print('Create path : {}'.format(fig_path))

    val_data_loader,train_data_loader = get_loader(mode=args.mode,
                             data_path = args.data_path,
                             input_path =args.input_path,
                             target_path = args.target_path,
                             mask_path=args.mask_path,
                             patch_n=args.patch_n,
                             patch_size=args.patch_size,
                             transform=args.transform,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    solver = Solver(args, val_data_loader,train_data_loader)
    print(args.input_path)
    if args.mode == 'train':
        solver.train()            ## 只训练加噪声的数据
    elif args.mode == 'test':
        # 2. only put add noise img through NET
        solver.test()             ## 只测试加噪声的数据
    elif args.mode == 'testori':  ## 所有数据全测试
        # 1. all test img through NET (）
        solver.testall()
    elif args.mode == 'testimg':  ## 存疑
    # 1. all test img through NET
        solver.testimg()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #######-----------------------For train----------------------------------------------    ## 只涉及train
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"#quebao program and pkgs cuda yizhi
    os.environ["CUDA_VISIBLE_DEVICES"]="0"#0 1
    parser.add_argument('--num_epochs', type=int, default=2)#default=150
    #propotion:loss = propotion*global_loss+(1-propotion)*local_loss
    parser.add_argument('--propotion', type=float, default='1',help='[0,1]')     ## 设计一个loss，1是没用上
    ####GGse_flag=1,startGGse
    parser.add_argument('--GGse_flag', type=int, default=1,help='0or1')          ## 有么有用本文设计的到模块，1是用到
########-----------------------------all-----------------------------------------------------  ### train test都涉及
    parser.add_argument('--input_path', type=str, default='NUDT-SIRST-Noise_v3/images')
    parser.add_argument('--mode', type=str, default='train',help='test;testori;train')  ##test测试加噪图片，testori测试所有图片
    parser.add_argument('--model', type=str, default='REDNet30', help='RED_CNN,REDNet30')     ### CNN 是10层  30是30层，论文里用的是30
    parser.add_argument('--patch_size', type=int, default=65, help='256;129;65;33;17;9;5;3')  ### 论文用的256
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./save/NUDT-SIRST-Noise_v3')
    parser.add_argument('--type_path', type=str, default='NUDT-SIRST-Noise_v3')
   #####------------------------------For test----------------------------------------------  ### 只涉及到test
    parser.add_argument('--img_save_path', type=str, default='./results/forall')    ## 存储测试数据降噪后图片以及对应图片质量信息
    parser.add_argument('--loadtime', type=str, default='2022_05_01_23_22_07')
    # train_img_flag='0',dont save denoising train_imgs
    # ... =1, save all train imgs throuh denoise_net
    parser.add_argument('--train_img_flag', type=int, default=0)                    ## inference的时候要不要把trainset纳入进来
    # -----------------------------------------------------------------------------------------

    parser.add_argument('--patch_n',  type=int, default=None)
    parser.add_argument('--data_path', type=str, default='D:/paper_project/IRDnNet/dataset')
    # x=noise_img=input_img,y=clean_img= target_img
    parser.add_argument('--target_path', type=str, default='NUDT-SIRST/images')
    parser.add_argument('--mask_path', type=str, default='NUDT-SIRST/mask0_1')    ### 分割的GT，最后没有用
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    #data_range = self.trunc_max - self.trunc_min
    parser.add_argument('--trunc_min', type=float, default=-16.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=20)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    args = parser.parse_args()
    main(args)
