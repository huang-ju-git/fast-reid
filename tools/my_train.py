# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.data import get_train_dataloader, get_test_dataloader
from fastreid.evaluation.testing import flatten_results_dict
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.modeling import build_model
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.evaluation import inference_on_dataset, print_csv_format, ReidEvaluator
from fastreid.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from fastreid.utils import comm
from fastreid.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)
from fastreid.utils.logger import setup_logger
from fastreid.modeling.losses import *

logger = setup_logger(output="msmt17_verif_log")
os.environ['CUDA_VISIBLE_DEVICES']='1'

# test的时候不是进行二分类，而是检查多分类的结果
def evaluate_on_dataset(cfg, model):
    total = 0
    correct = 0
    img_correct=0
    img_total=0
    with torch.no_grad():
        data_loader=get_test_dataloader()
        accuracy_positive=0
        count=0
        for idx, inputs in enumerate(data_loader):
            inputs=[term.cuda() for term in inputs]
            # outputs = model(inputs) #一个batch对应的outputs
            # print(outputs)
            # _, predicted = torch.max(outputs.data, dim=1)  # 取出每一行输出数据中最大值和最大值的下标   ，dim从上往下的行数是第0个维度，1是第二个维度，代表一行的所有数据
            # total += labels.size(0) # 一个批量样本的个数(batch_size)，labels就是属于0-9分类的向量
            # correct += (predicted == labels).sum().item()  # 猜对了的量

            bank_outputs, bank_targets=model(inputs)
            cls_outputs=bank_outputs['cls_outputs']
            prob, predicted = torch.max(cls_outputs.data, dim=1)  # 取出每一行输出数据中最大值和最大值的下标
            total += bank_targets.size(0)
            correct += (predicted == bank_targets).sum().item()

            # img_num=len(bank_targets)/cfg.DATALOADER.PERSON_NUMBER_TEST
            # #print(img_num)
            # for i in range(int(img_num)): #对于每一张图片
            #     img_total+=1
            #     for j in range(cfg.DATALOADER.PERSON_NUMBER_TEST): #对于和每个平均特征的特征差
            #         id=i*cfg.DATALOADER.PERSON_NUMBER_TEST+j
            #         if cls_outputs[id][1]>cls_outputs[id][0] and bank_targets[id]==1 :
            #             img_correct+=1
            
            bank_cls_outputs = bank_outputs['cls_outputs']
            positive_index=[i for i in range(len(bank_targets)) if bank_targets[i]==1]
            bank_cls_outputs_positive=[bank_cls_outputs[i] for i in positive_index ]
            bank_targets_positive=[bank_targets[i] for i in positive_index]
            bank_cls_outputs_positive = torch.stack(bank_cls_outputs_positive).cuda()
            bank_targets_positive = torch.stack(bank_targets_positive).cuda()
            accuracy_pos=log_accuracy(bank_cls_outputs_positive, bank_targets_positive)
            accuracy_positive+=accuracy_pos*len(bank_outputs)
            count+=len(bank_outputs)

        accuracy_positive=accuracy_positive/count
        logger.info("testing accuracy_positive is {}".format(accuracy_positive))
    logger.info('accuracy on test set: %f %%' % (100 * correct / total))
    # logger.info('img_correct on test set: %d ' % (img_correct))
    # logger.info('img_accuracy on test set: %f %%' % (100 * img_correct / img_total))
    results = OrderedDict()
    results["accuracy"]=100*correct/total
    # results["img_correct"]=img_correct
    # results["img_accuracy"]=100*img_correct/img_total
    results["accuracy_positive"]=accuracy_positive
    return results




def do_test(cfg, model):
    return evaluate_on_dataset(cfg, model)
    # results = OrderedDict()
    # for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
    #     logger.info("Prepare testing set")
    #     try:
    #         data_loader, evaluator = get_evaluator(cfg, dataset_name)
    #     except NotImplementedError:
    #         logger.warn(
    #             "No evaluator found. implement its `build_evaluator` method."
    #         )
    #         results[dataset_name] = {}
    #         continue
    #     results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED)
    #     results[dataset_name] = results_i

    #     if comm.is_main_process():
    #         assert isinstance(
    #             results, dict
    #         ), "Evaluator must return a dict on the main process. Got {} instead.".format(
    #             results
    #         )
    #         logger.info("Evaluation results for {} in csv format:".format(dataset_name))
    #         results_i['dataset'] = dataset_name
    #         print_csv_format(results_i)

    # if len(results) == 1:
    #     results = list(results.values())[0]

    # return results


def do_train(cfg, model, resume=False):
    data_loader = get_train_dataloader()
    data_loader_iter = iter(data_loader)

    model.train()
    optimizer = build_optimizer(cfg, model)

    iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    scheduler = build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    checkpointer = Checkpointer(
        model,
        cfg.OUTPUT_DIR,
        save_to_disk=comm.is_main_process(),
        optimizer=optimizer,
        **scheduler
    )

    start_epoch = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("epoch", -1) + 1
    )
    iteration = start_iter = start_epoch * iters_per_epoch

    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_iter = max_epoch * iters_per_epoch
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    delay_epochs = cfg.SOLVER.DELAY_EPOCHS

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_epoch)
    if len(cfg.DATASETS.TESTS) == 1:
        metric_name = "metric"
    else:
        metric_name = cfg.DATASETS.TESTS[0] + "/metric"

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR)
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support some hooks, such as
    # accurate timing, FP16 training and precise BN here,
    # because they are not trivial to implement in a small training loop
    logger.info("Start training from epoch {}".format(start_epoch))
    with EventStorage(start_iter) as storage:
        for epoch in range(start_epoch, max_epoch):
            storage.epoch = epoch
            accuracy_positive=0
            count=0
            for idx, data in enumerate(data_loader):
                #print(len(data[0])) #输出32
                # data = next(data_loader_iter)
                storage.iter = iteration

                data=[term.cuda() for term in data]
                outputs = model(data)

                bank_cls_outputs = outputs["bank_outputs"]['cls_outputs']
                bank_targets  = outputs["bank_targets"]
                positive_index=[i for i in range(len(bank_targets)) if bank_targets[i]==1]
                bank_cls_outputs_positive=[bank_cls_outputs[i] for i in positive_index ]
                bank_targets_positive=[bank_targets[i] for i in positive_index]
                bank_cls_outputs_positive = torch.stack(bank_cls_outputs_positive).cuda()
                bank_targets_positive = torch.stack(bank_targets_positive).cuda()
                accuracy_pos=log_accuracy(bank_cls_outputs_positive, bank_targets_positive)
                accuracy_positive+=accuracy_pos*len(data[0])
                count+=len(data[0])
                # logger.info("training accuracy_positive is {}".format(accuracy_positive))

                loss_dict=model.losses(outputs)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                if iteration - start_iter > 5 and \
                        ((iteration + 1) % 200 == 0 or iteration == max_iter - 1) and \
                        ((iteration + 1) % iters_per_epoch != 0):
                    for writer in writers:
                        writer.write()

                iteration += 1

                if iteration <= warmup_iters:
                    scheduler["warmup_sched"].step()

            accuracy_positive=accuracy_positive/count
            logger.info("training accuracy_positive is {}".format(accuracy_positive))
            # Write metrics after each epoch
            for writer in writers:
                writer.write()

            if iteration > warmup_iters and (epoch + 1) > delay_epochs:
                scheduler["lr_sched"].step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                model.eval()
                results = do_test(cfg, model)
                model.train()
                # Compared to "train_net.py", the test results are not dumped to EventStorage
            else:
                results = {}
            flatten_results = flatten_results_dict(results)

            metric_dict = dict(metric=flatten_results[metric_name] if metric_name in flatten_results else -1)
            periodic_checkpointer.step(epoch, **metric_dict)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

# python tools/my_train.py --config-file ./configs/BOT-verif-msmt17.yml --num-gpus 1