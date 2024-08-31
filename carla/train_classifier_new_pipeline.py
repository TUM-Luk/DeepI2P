import open3d
import time
import copy
import numpy as np
import math
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('Agg')

from models.multimodal_classifier import MMClassifer, MMClassiferCoarse
from data.carla_pc_img_pose_loader import make_carla_dataloader, CarlaLoader
import tqdm


if __name__=='__main__':
    from carla import options_0829_baseline as options
    opt = options.Options()
    datetime_str = time.strftime("_%m%d_%H%M", time.localtime())
    logdir = './runs_carla/'+str(opt.version)+datetime_str
    if os.path.isdir(logdir):
        user_answer = input("The log directory %s exists, do you want to delete it? (y or n) : " % logdir)
        if user_answer == 'y':
            # delete log folder
            shutil.rmtree(logdir)
        else:
            exit()
    else:
        os.makedirs(logdir)
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=logdir)

    # create train/val dataset & dataloader
    trainset, trainloader = make_carla_dataloader(mode='train', opt=opt)
    print('#training point clouds = %d' % len(trainset))
    valset, valloader = make_carla_dataloader(mode='val', opt=opt)
    print('#validating point clouds = %d' % len(valset))
    testset, testloader = make_carla_dataloader(mode='test', opt=opt)
    print('#testing point clouds = %d' % len(testset))
    
    # trainset = OxfordLoader(opt.dataroot, 'train', opt)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
    #                                           num_workers=opt.dataloader_threads, drop_last=True, pin_memory=True)
    # testset = OxfordLoader(opt.dataroot, 'val', opt)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
    #                                          num_workers=opt.dataloader_threads, pin_memory=True)

    # create model, optionally load pre-trained model
    if opt.is_fine_resolution:
        model = MMClassifer(opt, writer)
    else:
        model = MMClassiferCoarse(opt, writer)
    # model.load_model('/home/tohar/repos/point-img-feature/oxford/workspace/640x384-noCrop/checkpoints/best.pth')

    best_test_accuracy = 0
    for epoch in range(1, 101):
        len_trainloader = len(trainloader)
        epoch_iter = 0
        for i, data in enumerate(trainloader):
            # if i > 5:
            #     break
            pc, intensity, sn, node_a, node_b, \
            P, img, K, t_ij = data
            B = pc.size()[0]

            iter_start_time = time.time()
            epoch_iter += B
            model.global_step_inc(B)

            model.set_input(pc, intensity, sn, node_a, node_b,
                            P, img, K)
            model.optimize()

            if i % int(50) == 0 and i > 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / opt.batch_size
                train_loss_dict, test_loss_dict = model.get_current_errors()
                train_accuracy_dict, test_accuracy_dict = model.get_current_accuracy()
                print(f"Epoch {epoch}/101, ite {i}/{len_trainloader}")
                model.print_loss_dict(train_loss_dict, train_accuracy_dict, t)

                model.write_loss()
                model.write_accuracy()

                model.write_img()
                model.write_pc_label(model.train_visualization['pc'],
                                     model.train_visualization['coarse_labels'],
                                     'coarse_labels')
                model.write_pc_label(model.train_visualization['pc'],
                                     model.train_visualization['coarse_predictions'],
                                     'coarse_predictions')
                if opt.is_fine_resolution:
                    model.write_classification_visualization(model.train_visualization['KP_pc_pxpy'],
                                                             model.train_visualization['coarse_predictions'],
                                                             model.train_visualization['fine_predictions'],
                                                             model.train_visualization['coarse_labels'],
                                                             model.train_visualization['fine_labels'],
                                                             t_ij)
                else:
                    model.write_classification_visualization(model.train_visualization['KP_pc_pxpy'],
                                                             model.train_visualization['coarse_predictions'],
                                                             model.train_visualization['coarse_labels'],
                                                             t_ij)

        # EPOCH DONE
        
        # ------------------------- test on seen region (valset) ----------------------------
        test_start_time = time.time()
        test_batch_sum = 0
        test_loss_sum = {'loss': 0, 'coarse': 0, 'fine': 0}
        test_accuracy_sum = {'coarse_accuracy': 0, 'fine_accuracy': 0}
        for i, data in tqdm.tqdm(enumerate(valloader), total=len(valloader)):
            # if i > 5:
            #     break
            pc, intensity, sn, node_a, node_b, \
            P, img, K, t_ij = data
            B = pc.size()[0]

            model.set_input(pc, intensity, sn, node_a, node_b,
                            P, img, K)

            model.test_model()
            _, test_loss_dict = model.get_current_errors()
            _, test_accuracy = model.get_current_accuracy()

            test_batch_sum += B
            test_loss_sum['loss'] += B * test_loss_dict['loss']
            test_loss_sum['coarse'] += B * test_loss_dict['coarse']
            test_accuracy_sum['coarse_accuracy'] += B * test_accuracy['coarse_accuracy']
            if opt.is_fine_resolution:
                test_loss_sum['fine'] += B * test_loss_dict['fine']
                test_accuracy_sum['fine_accuracy'] += B * test_accuracy['fine_accuracy']

        test_loss_sum['loss'] /= test_batch_sum
        test_loss_sum['coarse'] /= test_batch_sum
        test_accuracy_sum['coarse_accuracy'] /= test_batch_sum
        if opt.is_fine_resolution:
            test_loss_sum['fine'] /= test_batch_sum
            test_accuracy_sum['fine_accuracy'] /= test_batch_sum
        test_persample_time = (time.time() - test_start_time) / test_batch_sum

        print('Test loss and accuracy:')
        model.print_loss_dict(test_loss_sum, test_accuracy_sum, test_persample_time)

        model.writer.add_scalars('test_loss (seen region)',
                                test_loss_sum,
                                global_step=model.global_step)
        model.writer.add_scalars('test_accuracy (seen region)',
                                test_accuracy_sum,
                                global_step=model.global_step)
        
        # ----------------- test on unseen region (testset) -------------------------------
        test_start_time = time.time()
        test_batch_sum = 0
        test_loss_sum = {'loss': 0, 'coarse': 0, 'fine': 0}
        test_accuracy_sum = {'coarse_accuracy': 0, 'fine_accuracy': 0}
        for i, data in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            # if i > 5:
            #     break
            pc, intensity, sn, node_a, node_b, \
            P, img, K, t_ij = data
            B = pc.size()[0]

            model.set_input(pc, intensity, sn, node_a, node_b,
                            P, img, K)

            model.test_model()
            _, test_loss_dict = model.get_current_errors()
            _, test_accuracy = model.get_current_accuracy()

            test_batch_sum += B
            test_loss_sum['loss'] += B * test_loss_dict['loss']
            test_loss_sum['coarse'] += B * test_loss_dict['coarse']
            test_accuracy_sum['coarse_accuracy'] += B * test_accuracy['coarse_accuracy']
            if opt.is_fine_resolution:
                test_loss_sum['fine'] += B * test_loss_dict['fine']
                test_accuracy_sum['fine_accuracy'] += B * test_accuracy['fine_accuracy']

        test_loss_sum['loss'] /= test_batch_sum
        test_loss_sum['coarse'] /= test_batch_sum
        test_accuracy_sum['coarse_accuracy'] /= test_batch_sum
        if opt.is_fine_resolution:
            test_loss_sum['fine'] /= test_batch_sum
            test_accuracy_sum['fine_accuracy'] /= test_batch_sum
        test_persample_time = (time.time() - test_start_time) / test_batch_sum

        print('Test loss and accuracy:')
        model.print_loss_dict(test_loss_sum, test_accuracy_sum, test_persample_time)

        model.writer.add_scalars('test_loss (unseen region)',
                                test_loss_sum,
                                global_step=model.global_step)
        model.writer.add_scalars('test_accuracy (unseen region)',
                                test_accuracy_sum,
                                global_step=model.global_step)
        
        # record best test loss
        if test_accuracy_sum['coarse_accuracy'] > best_test_accuracy:
            best_test_accuracy = test_accuracy_sum['coarse_accuracy']
            print('--- best test coarse accuracy %f' % best_test_accuracy)

        print('Epoch %d done.' % epoch)

        if epoch % opt.lr_decay_step == 0 and epoch > 0:
            model.update_learning_rate(opt.lr_decay_scale)

        # save network
        if epoch % 5 == 0 and epoch>0:
            print("Saving network...")
            model.save_network(model.detector, "v%s-gpu%d-epoch%d-%f.pth" % (opt.version,
                                                                             opt.gpu_ids[0],
                                                                             epoch,
                                                                             test_accuracy_sum['coarse_accuracy']))





