import argparse
import gc
import os
import os.path as osp
import threading
import time
import cv2
import torch
import json

import numpy as np
import mini
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
import sys
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class Context:
    def __init__(self):
        self.close = False #关闭程序
        self.mode=0
        #0 井下 1 井上
        self.exp=None
        self.model=None
        self.output='.'
        self.predictor=None
        self.ready=False
        self.pic='.'
        self.args=None
        self.modelPool=[1,2,3]
    def config_load(self):
        if not self.ready:
            logger.info("等待模型加载...")
        else:
            if self.exp is not None:
                output = 'a' if self.mode == 1 else 'd'
                conf = get_confidence(f'{output}config.json')
                if self.exp.test_conf !=conf:
                    self.exp.test_conf = conf
                    self.predictor = Predictor(self.model, self.exp, None, None, "cuda", True)
                    logger.info(f"{'井上' if self.mode else '井下'}模型confidence调整到:{self.exp.test_conf}")
    def load(self):
        self.ready=False
        logger.info(f"load model{'井上' if self.mode else '井下'}")
        # self.model, self.exp, self.predictor, self.output, self.pic=self.modelPool[self.mode]
        if self.exp is not None:
            self.exp.delete()
        output='a' if self.mode==1 else 'd'
        self.model, self.exp, self.predictor, self.output, self.pic=main(context.__getattribute__(f"exp{output}"), self.args, output)
        self.exp.test_conf = get_confidence(f'{output}config.json')
        logger.info(f"{'井上' if self.mode else '井下'}模型confidence:{self.exp.test_conf}")
        self.predictor=Predictor(self.model, self.exp, None, None, "cuda", True)
        self.ready=True
        logger.info("加载成功")
        logger.info("Awaiting frames...")
    def toggle(self):
        # with self.lock:  # 使用锁保护共享资源
        # self.exit()
        self.mode=1-self.mode
        if not self.ready:
            logger.info("等待模型加载...")
        else:
            logger.info(f"切换模型为{'井上' if self.mode else '井下'}")
            self.load()
    def exit(self):
        del  self.model
        del self.predictor
        torch.cuda.empty_cache()
# pyinstaller --onefile --name=Anti-CollisionDetector Anti-collisionDetector.py
BASE_PATH='.'
if getattr(sys, 'frozen', False):
    # 如果是打包后的可执行文件
    BASE_PATH = sys._MEIPASS
else:
    # 如果是开发环境下运行
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
def read_config_path():
    if getattr(sys, 'frozen', False):
        # 如果是打包后的可执行文件
        base_path = sys._MEIPASS
    else:
        # 如果是开发环境下运行
        base_path = os.path.dirname(os.path.abspath(__file__))
    # 打印根目录下的所有文件和文件夹
    print(f"根目录 {base_path} 下的文件和文件夹:")
    try:
        items = os.listdir(base_path)
        for item in items:
            print(item)
    except FileNotFoundError:
        print(f"未找到根目录: {base_path}")
    config_path = os.path.join(base_path, 'config.json')
    print(f"尝试访问的配置文件路径: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"未找到配置文件: {config_path}")
        return None
    except json.JSONDecodeError:
        print("配置文件格式错误")
        return None

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./pic", help="path to images or video"
        # "--path",default="./videos/test.mp4",help="path to images or video"

    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-fd",
        "--exp_filed",
        default= os.path.join(BASE_PATH,"./exps/example/mot/yolox_x_light.py"),
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument(
        "-fa",
        "--exp_filea",
        default=os.path.join(BASE_PATH, "./exps/example/mot/yolox_x_mix_det.py"),
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-cd", "--ckptd", default="./mt/best_ckpt.pth.tar", type=str,
                        help="ckpt for eval")
    parser.add_argument("-ca", "--ckpta", default="./mt/bytetrack_x_mot17.pths.tar", type=str,
                        help="ckpt for eval")
    # parser.add_argument("-cd", "--ckptd", default=os.path.join(BASE_PATH,"./mt/best_ckpt.pth.tar"), type=str,
    #                     help="ckpt for eval")
    # parser.add_argument("-ca", "--ckpta", default=os.path.join(BASE_PATH, "./mt/bytetrack_x_mot17.pths.tar"), type=str,
    #                     help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.6, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=True,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class Predictor(object):
    def __init__(
            self,
            model,
            # 模型
            exp,
            # ./exps/example/mot/yolox_x_mix_det.py中的EXP类
            trt_file=None,
            decoder=None,
            device=torch.device("cpu"),
            # gpu
            fp16=False
    #         True
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        # 识别是否是人故为1
        self.confthre = exp.test_conf
        # logger.info(f"confidence:{self.confthre}")
        # 置信度
        self.nmsthre = exp.nmsthre
        # 最大限制阈值
        self.test_size = exp.test_size
        # 测试大小
        self.device = device
        self.fp16 = fp16
        # True
        if trt_file is not None:
            # 默认关闭
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        # 原始大小
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        # 图像预处理，图像，目标大小，
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            # img = img.half()  # to FP16  想用gpu就这行
            img = img.float()  # 想用cpu就这行

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            # 后处理
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

def read_mode_file(context):
    ttd=0
    while True:
        ttd += 1
        if ttd % 2 == 0:
            context.config_load()
            ttd=0
        try:
            with open('mode.txt', 'r') as file:
                content = file.read().strip()
                if content in ['0', '1']:
                    mode = int(content)
                    if mode != context.mode:
                        # context.mode = mode
                        context.toggle()
        except FileNotFoundError:
            print("mode.txt file not found.")
        except ValueError:
            print("Invalid content in mode.txt.")

        time.sleep(1)
def create_folders_if_not_exist(pic_folder, rec_folder):
    if not osp.isdir(pic_folder):
        os.makedirs(pic_folder)
    if not osp.isdir(rec_folder):
        os.makedirs(rec_folder)


def convert_float32_to_float(obj):
    if isinstance(obj, dict):
        return {k: convert_float32_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(element) for element in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj


def image_demo(args):
    timer = Timer()
    # 自定义timer
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    # 时间帧

    mode=context.mode
    # pic_folder=apic_folder if mode==1 else dpic_folder
    while True:
        if context.close:
            break
        else:
            time.sleep(0.1)
        try:
            # 死循环
            while True:
                if context.close:
                    break
                if not context.ready:
                    time.sleep(0.5)
                    continue
                files = sorted([f for f in os.listdir(context.pic) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp'))])

                if not files:
                    time.sleep(0.1)  # 这个休眠时间可以根据需要调整
                    continue
                mode=context.mode
                logger.info(f"Found image: {files}")
                for file_name in files:
                    tracker = BYTETracker(args, frame_rate=args.fps)
                    img_path = osp.join(context.pic, file_name)
                    img_basename = osp.basename(img_path)
                    img_basename_no_ext = osp.splitext(img_basename)[0]
                    try:
                        # 读取图像并执行预测
                        if mode!=context.mode:
                            break
                        img = cv2.imread(img_path)
                        if img is None:
                            logger.info(f"[!] Image {img_path} permission denied. Process in next iteration.")
                            continue  # 跳过当前图像，继续处理下一个
                        if mode!=context.mode:
                            break
                        outputs, img_info = context.predictor.inference(img_path, timer)
                        #预测
                        logger.info(f"Image {img_path} processed by model.")
                        predictions = []
                        if mode!=context.mode:
                            break
                        if outputs[0] is not None:
                            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']],
                                                            context.exp.test_size)
                            if mode != context.mode:
                                break
                            for t in online_targets:
                                tlwh = t.tlwh
                                tid = t.track_id
                                score = t.score
                                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                                    centroid_x = tlwh[0] + tlwh[2] / 2
                                    centroid_y = tlwh[1] + tlwh[3] / 2
                                    predictions.append({
                                        'id': tid,
                                        'bbox': [tlwh[0], tlwh[1], tlwh[2], tlwh[3]],
                                        'centroid': [centroid_x, centroid_y],
                                        'confidence': score,
                                    })

                            # 保存预测后的图像到rec文件夹
                        #     rec_img_path = osp.join(context.output, img_basename)
                        #     cv2.imwrite(rec_img_path, plot_tracking(img_info['raw_img'], [t.tlwh for t in online_targets],
                        #                                                 [t.track_id for t in online_targets]))
                        # else:
                        #     rec_img_path = osp.join(context.output, img_basename)
                        #     cv2.imwrite(rec_img_path, img)

                        # 保存预测结果为JSON文件
                        rec_json_path = osp.join(context.output, f"{img_basename_no_ext}.json")
                        with open(rec_json_path, 'w') as f:
                            json.dump(convert_float32_to_float(predictions), f, indent=4)

                        # 删除原始图像
                        os.remove(img_path)
                        logger.info(f"Processed and saved {rec_json_path}. Original image is deleted.")

                    except Exception as e:
                        logger.warning(f"Failed to process image {img_path} in this iteration: {e}. Image has been deleted.")
                        os.remove(img_path)

                # 等待一段时间再检查
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Program interrupted by user.")


def main(exp, args,output):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = os.path.join('.', f"{output}Rec")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pic_dir= os.path.join('.', f"{output}Pic")
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    # output_dir = os.path.join('.', 'dRec')
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    if args.trt:
        # 默认False没有用到trt加速
        args.device = "gpu"
    args.device = "gpu"
    device="cuda" if (args.device == "gpu" ) else "cpu"
    # print(args.device,device,"cuda")
    print(torch.cuda.is_available())
    args.device = torch.device(device)
    print("使用的是", args.device)
    # logger.info("Args: {}".format(args))


    exp.test_conf = get_confidence(f'{output}config.json')
    #     置信度
    if args.nms is not None:
        # 空的
        exp.nmsthre = args.nms
    #     非极大值抑制（Non-Maximum Suppression，NMS） 的阈值。NMS 是目标检测算法中常用的一种方法，用于从多个候选框中选择最合适的框，避免重复检测同一个目标。
    if args.tsize is not None:
        # 默认空的
        exp.test_size = (args.tsize, args.tsize)

    # test_size 可能是图像在测试阶段的尺寸大小，例如输入图像需要缩放到的尺寸。
    model_c = exp.get_model()
    model=model_c.to(args.device)
    del model_c
    #
    # gc.collect()
    # print(next(model.backbone.parameters()).device)  # 应该输出 cuda:0
    # print(next(model.head.parameters()).device)  # 应该输出 cuda:0
    # exp.testModel()

    # 加载模型
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    model.eval()

    if not args.trt:
        # 不trt加速的话加载模型参数
        if "a" in output:
            ckpt_file = args.ckpta if args.ckpta else osp.join(output_dir, "best_ckpt.pth.tar")
            print(args.ckpta,"NoneA")
        elif "d" in output:
            ckpt_file = args.ckptd if args.ckptd else osp.join(output_dir, "best_ckpt.pth.tar")
            print(args.ckptd, "NoneD")
        logger.info("Loading model...")
        ckpt = torch.load(ckpt_file, map_location="cpu")

        model.load_state_dict(ckpt["model"])

        del ckpt
        ckpt=None
        gc.collect()
        logger.info("Model Loaded.")

    if args.fuse:
        logger.info("Fusing model...")
        # 默认True
        model = fuse_model(model)
    #     将模型中的卷积层（conv）和批归一化层（bn）融合，以优化模型的推理速度，减少计算开销。

    if args.fp16:
        model = model.float()
    # 转成float16 但是转成了32是个失误
    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(trt_file), "TensorRT model is not found!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    # exp.makeModel(model)
    return  model,exp,Predictor(model, exp, trt_file, decoder, args.device, args.fp16),output_dir,pic_dir


def get_confidence(config_file_path):
    if os.path.exists(config_file_path):
        try:
            # 打开并读取 JSON 文件
            with open(config_file_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
                # 从配置中获取 confidence 值
                confidence = config.get('confidence')
                if confidence is not None:
                    return confidence
                else:
                    print("配置文件中未找到 'confidence' 字段，将使用默认值。")
        except json.JSONDecodeError:
            print("配置文件格式错误，将使用默认值。")
    else:
        print("配置文件不存在，将创建默认配置文件。")
    # 如果文件不存在、格式错误或未找到 'confidence' 字段，使用默认值
    default_config = {"confidence": 0.6}
    try:
        # 创建并写入默认配置文件
        with open(config_file_path, 'w', encoding='utf-8') as file:
            json.dump(default_config, file, indent=4)
    except Exception as e:
        print(f"创建配置文件时出错: {e}")
    return 0.6
def read_config(file_path):
    with open(file_path, 'r') as f:
        cfg = json.load(f)
    return cfg
def anti_main_():
    global  config
    # config = read_config_path()
    # 置信度0.6
    args = make_parser().parse_args()
    # Namespace(demo='image', experiment_name=None, name=None, path='./pic', camid=0, save_result=True, exp_file='./exps/example/mot/yolox_x_mix_det.py', ckpt='./pretrained/bytetrack_x_mot17.pths.tar', device='gpu', conf=0.6, nms=None, tsize=None, fps=30, fp16=True, fuse=True, trt=False, track_thresh=0.5, track_buffer=30, match_thresh=0.8, aspect_ratio_thresh=1.6, min_box_area=10, mot20=False)
    print(args)
    context.args=args
    context.expa = get_exp(args.exp_filea, args.name)
    context.expd = get_exp(args.exp_filed, args.name)
    # context.mode=1

    # self.model, self.exp, self.predictor, self.output, self.pic = main(context.expd, self.args, 'd')
    bg=time.time()
    # context.modelPool.append(main(context.expd, args,'d'))
    # context.modelPool.append(main(context.expa, args,'a'))
    pt="井下" if context.mode==0 else "井上"
    context.load()
    image_demo( args)
    # print("结束")

if __name__ == "__main__":

    if not os.path.exists('mode.txt'):
        try:
            with open('mode.txt', 'w') as file:
                file.write('0')
            print("mode.txt file created with initial value 0.")
        except Exception as e:
            print(f"Error creating mode.txt: {e}")
    # 创建 tkinter 主窗口
    # 创建一个事件，用于线程间通信
    # read_config_path()
    event = threading.Event()
    context = Context()
    # 启动 tkinter 窗口的线程
    # context.args=args
    main_window = threading.Thread(target=mini.create_main_window, args=(event, logger, context), daemon=True)
    main_window.start()

    # 等待窗口创建完成
    event.wait()

    # 启动 main 函数
    threading.Thread(target=anti_main_, daemon=True).start()
    thread = threading.Thread(target=read_mode_file, args=(context,))
    thread.daemon = True  # 设置为守护线程，主程序退出时线程也会退出
    thread.start()
    main_window.join()
    print("窗口退出")
