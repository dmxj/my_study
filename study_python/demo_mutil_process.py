#encoding:utf-8
from multiprocessing import Pool,Manager,Process,Queue
import os
import time
import random
import torch
import torchvision
import pretrainedmodels
import pretrainedmodels.utils as utils
import cv2
import numpy as np

def long_time_task(ll,name,lock):
    with lock:
        print('Run task %s (%s)...' % (name, os.getpid()))
        start = time.time()
        time.sleep(random.random() * 3)
        end = time.time()
        ll.append(random.randint(0,10))
        print("task {} sleep {}".format(name,end - start))

def test_process_pool():
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    with Manager() as manager:
        ll = manager.list()
        lock = manager.Lock()
        for i in range(5):
            p.apply_async(long_time_task, args=(ll,i,lock,))
            # print("result is : [task {} sleep {}]".format(i,res))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(ll)

resnet_model_path = "/Users/rensike/Resources/models/pytorch/resnet18-5c106cde.pth"
def infer_process(my_queue,img_input,model):
    print("enter infer process")
    output = model(img_input)
    _, predicted = torch.max(output.data, 1)
    print("model predict result is:",predicted.numpy()[0])
    my_queue.put(predicted.numpy()[0])
    # pred_list.append(predicted.numpy()[0])

def infer_callback(predict_res):
    print("get predict result:",predict_res)

def test_multi_process_infer():
    model = pretrainedmodels.__dict__["resnet18"](num_classes=1000, pretrained='imagenet')
    tf_img = utils.TransformImage(model)
    load_img = utils.LoadImage()
    img_list = ["010.jpg","004.jpg","005.jpg","011.jpg","012.jpg","boy.jpg"]
    res = {}
    t0 = time.time()
    p_list = []
    my_queue = Queue()
    with Manager() as manager:
        pred_list = manager.list()
        my_pool = Pool(4)
        for img in img_list:
            input_img = load_img(img)
            input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
            input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
            input_img = torch.autograd.Variable(input_tensor,requires_grad=False)
            print("image:",img)
            p = Process(target=infer_process, args=(my_queue,input_img,model))
            p.start()
            p_list.append(p)
            # p.start()
            # p.join()
            # xx = my_pool.apply_async(infer_process,args=(pred_list,input_img,model,),callback=infer_callback)
            # res[img] = xx
        # my_pool.close()
        # my_pool.join()
    for p in p_list:
        p.join()

    while not my_queue.empty():
        value = my_queue.get(True)
        print("queue get a predict result:",value)
        # time.sleep(random.random())

    print("Time cost:",time.time() - t0)
    print("模型推理完毕...")
    # print(pred_list)
        # for im in res:
        #     print(res[im].get())

def test_batch_infer():
    model = pretrainedmodels.__dict__["resnet18"](num_classes=1000, pretrained='imagenet')
    tf_img = utils.TransformImage(model)
    load_img = utils.LoadImage()
    img_list = ["010.jpg","004.jpg","005.jpg","011.jpg","012.jpg","boy.jpg"]
    t0 = time.time()
    for img in img_list:
        input_img = load_img(img)
        input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
        input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
        input_img = torch.autograd.Variable(input_tensor,requires_grad=False)
        output = model(input_img)
        _, predicted = torch.max(output.data, 1)
        print("model predict result is:",predicted.numpy()[0])
    print("Time cost:",time.time() - t0)

if __name__ == "__main__":
    # model = torchvision.models.resnet18()
    # model.load_state_dict(torch.load(resnet_model_path))
    # model.eval()

    # model = pretrainedmodels.__dict__["resnet18"](num_classes=1000, pretrained='imagenet')
    # model.eval()
    # tf_img = utils.TransformImage(model)
    # load_img = utils.LoadImage()
    # input_img = load_img("004.jpg")
    # input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
    # input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
    # input_img = torch.autograd.Variable(input_tensor,requires_grad=False)
    # output = model(input_img)
    # _, predicted = torch.max(output.data, 1)
    # print("model predict result is:",predicted.numpy()[0])

    # test_process_pool()

    # test_batch_infer()

    test_multi_process_infer()

    # with Manager() as manager:
    #     box_list = manager.list()
    #     box_list.append(33)
    #     print(box_list)