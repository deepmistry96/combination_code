#YAMSpy imports
import time
import curses
from collections import deque
from itertools import cycle
from yamspy import MSPy


#YAMS VARS
import subprocess
import sys
# proc = subprocess.Popen('python -m serial.tools.list_ports', shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# proc = subprocess.Popen( "python", "-m", "serial.tools.list_ports", shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# comports= proc.stdout.readlines()


import os 
comports = os.system('python -m serial.tools.list_ports') 
print(comports)

SERIAL_PORT = "/dev/cu.usbmodem3874346A32321"
port_input = input("\n\nUsing " + SERIAL_PORT + "\nPaste the comport that you want to use. Enter nothing to use the provided\n")
if (port_input != ""):
    SERIAL_PORT = port_input

print("Using the following comport:" + SERIAL_PORT)

#YAMSPY gui vars
# Max periods for:
CTRL_LOOP_TIME = 1/100
SLOW_MSGS_LOOP_TIME = 1/5 # these messages take a lot of time slowing down the loop...
NO_OF_CYCLES_AVERAGE_GUI_TIME = 10





#Function defs

#YAMS

def run_curses(external_function):
    result=1

    try:
        # get the curses screen window
        screen = curses.initscr()

        # turn off input echoing
        curses.noecho()

        # respond to keys immediately (don't wait for enter)
        curses.cbreak()

        # non-blocking
        screen.timeout(0)

        # map arrow keys to special values
        screen.keypad(True)

        screen.addstr(1, 0, "Press 'q' to quit, 'r' to reboot, 'm' to change mode, 'a' to arm, 'd' to disarm and arrow keys to control", curses.A_BOLD)
        
        result = external_function(screen)

    finally:
        # shut down cleanly
        curses.nocbreak(); screen.keypad(0); curses.echo()
        curses.endwin()
        if result==1:
            print("An error occurred... probably the serial port is not available ;)")

def keyboard_controller(screen):

    CMDS = {
            'roll':     1500,
            'pitch':    1500,
            'throttle': 900,
            'yaw':      1500,
            'aux1':     1000,
            'aux2':     1000
            }

    # This order is the important bit: it will depend on how your flight controller is configured.
    # Below it is considering the flight controller is set to use AETR.
    # The names here don't really matter, they just need to match what is used for the CMDS dictionary.
    # In the documentation, iNAV uses CH5, CH6, etc while Betaflight goes AUX1, AUX2...
    CMDS_ORDER = ['roll', 'pitch', 'throttle', 'yaw', 'aux1', 'aux2']

    # "print" doesn't work with curses, use addstr instead
    try:
        screen.addstr(15, 0, "Connecting to the FC...")

        with MSPy(device=SERIAL_PORT, loglevel='WARNING', baudrate=115200) as board:
            if board == 1: # an error occurred...
                return 1

            screen.addstr(15, 0, "Connecting to the FC... connected!")
            screen.clrtoeol()
            screen.move(1,0)

            average_cycle = deque([0]*NO_OF_CYCLES_AVERAGE_GUI_TIME)

            # It's necessary to send some messages or the RX failsafe will be activated
            # and it will not be possible to arm.
            command_list = ['MSP_API_VERSION', 'MSP_FC_VARIANT', 'MSP_FC_VERSION', 'MSP_BUILD_INFO', 
                            'MSP_BOARD_INFO', 'MSP_UID', 'MSP_ACC_TRIM', 'MSP_NAME', 'MSP_STATUS', 'MSP_STATUS_EX',
                            'MSP_BATTERY_CONFIG', 'MSP_BATTERY_STATE', 'MSP_BOXNAMES']

            if board.INAV:
                command_list.append('MSPV2_INAV_ANALOG')
                command_list.append('MSP_VOLTAGE_METER_CONFIG')

            for msg in command_list: 
                if board.send_RAW_msg(MSPy.MSPCodes[msg], data=[]):
                    dataHandler = board.receive_msg()
                    board.process_recv_data(dataHandler)
            if board.INAV:
                cellCount = board.BATTERY_STATE['cellCount']
            else:
                cellCount = 0 # MSPV2_INAV_ANALOG is necessary
            min_voltage = board.BATTERY_CONFIG['vbatmincellvoltage']*cellCount
            warn_voltage = board.BATTERY_CONFIG['vbatwarningcellvoltage']*cellCount
            max_voltage = board.BATTERY_CONFIG['vbatmaxcellvoltage']*cellCount

            screen.addstr(15, 0, "apiVersion: {}".format(board.CONFIG['apiVersion']))
            screen.clrtoeol()
            screen.addstr(15, 50, "flightControllerIdentifier: {}".format(board.CONFIG['flightControllerIdentifier']))
            screen.addstr(16, 0, "flightControllerVersion: {}".format(board.CONFIG['flightControllerVersion']))
            screen.addstr(16, 50, "boardIdentifier: {}".format(board.CONFIG['boardIdentifier']))
            screen.addstr(17, 0, "boardName: {}".format(board.CONFIG['boardName']))
            screen.addstr(17, 50, "name: {}".format(board.CONFIG['name']))


            slow_msgs = cycle(['MSP_ANALOG', 'MSP_STATUS_EX', 'MSP_MOTOR', 'MSP_RC'])

            cursor_msg = ""
            last_loop_time = last_slow_msg_time = last_cycleTime = time.time()
            while True:
                start_time = time.time()

                char = screen.getch() # get keypress
                curses.flushinp() # flushes buffer
                

                #
                # Key input processing
                #

                #
                # KEYS (NO DELAYS)
                #
                if char == ord('q') or char == ord('Q'):
                    break

                elif char == ord('d') or char == ord('D'):
                    cursor_msg = 'Sending Disarm command...'
                    CMDS['aux1'] = 1000

                elif char == ord('r') or char == ord('R'):
                    screen.addstr(3, 0, 'Sending Reboot command...')
                    screen.clrtoeol()
                    board.reboot()
                    time.sleep(0.5)
                    break

                elif char == ord('a') or char == ord('A'):
                    cursor_msg = 'Sending Arm command...'
                    CMDS['aux1'] = 1800

                #
                # The code below is expecting the drone to have the
                # modes set accordingly since everything is hardcoded.
                #
                elif char == ord('m') or char == ord('M'):
                    if CMDS['aux2'] <= 1300:
                        cursor_msg = 'Horizon Mode...'
                        CMDS['aux2'] = 1500
                    elif 1700 > CMDS['aux2'] > 1300:
                        cursor_msg = 'Flip Mode...'
                        CMDS['aux2'] = 2000
                    elif CMDS['aux2'] >= 1700:
                        cursor_msg = 'Angle Mode...'
                        CMDS['aux2'] = 1000

                elif char == ord('w') or char == ord('W'):
                    CMDS['throttle'] = CMDS['throttle'] + 10 if CMDS['throttle'] + 10 <= 2000 else CMDS['throttle']
                    cursor_msg = 'W Key - throttle(+):{}'.format(CMDS['throttle'])

                elif char == ord('e') or char == ord('E'):
                    CMDS['throttle'] = CMDS['throttle'] - 10 if CMDS['throttle'] - 10 >= 1000 else CMDS['throttle']
                    cursor_msg = 'E Key - throttle(-):{}'.format(CMDS['throttle'])

                elif char == curses.KEY_RIGHT:
                    CMDS['roll'] = CMDS['roll'] + 10 if CMDS['roll'] + 10 <= 2000 else CMDS['roll']
                    cursor_msg = 'Right Key - roll(-):{}'.format(CMDS['roll'])

                elif char == curses.KEY_LEFT:
                    CMDS['roll'] = CMDS['roll'] - 10 if CMDS['roll'] - 10 >= 1000 else CMDS['roll']
                    cursor_msg = 'Left Key - roll(+):{}'.format(CMDS['roll'])

                elif char == curses.KEY_UP:
                    CMDS['pitch'] = CMDS['pitch'] + 10 if CMDS['pitch'] + 10 <= 2000 else CMDS['pitch']
                    cursor_msg = 'Up Key - pitch(+):{}'.format(CMDS['pitch'])

                elif char == curses.KEY_DOWN:
                    CMDS['pitch'] = CMDS['pitch'] - 10 if CMDS['pitch'] - 10 >= 1000 else CMDS['pitch']
                    cursor_msg = 'Down Key - pitch(-):{}'.format(CMDS['pitch'])

                #
                # IMPORTANT MESSAGES (CTRL_LOOP_TIME based)
                #
                if (time.time()-last_loop_time) >= CTRL_LOOP_TIME:
                    last_loop_time = time.time()
                    # Send the RC channel values to the FC
                    if board.send_RAW_RC([CMDS[ki] for ki in CMDS_ORDER]):
                        dataHandler = board.receive_msg()
                        board.process_recv_data(dataHandler)

                #
                # SLOW MSG processing (user GUI)
                #
                if (time.time()-last_slow_msg_time) >= SLOW_MSGS_LOOP_TIME:
                    last_slow_msg_time = time.time()

                    next_msg = next(slow_msgs) # circular list

                    # Read info from the FC
                    if board.send_RAW_msg(MSPy.MSPCodes[next_msg], data=[]):
                        dataHandler = board.receive_msg()
                        board.process_recv_data(dataHandler)
                        
                    if next_msg == 'MSP_ANALOG':
                        voltage = board.ANALOG['voltage']
                        voltage_msg = ""
                        if min_voltage < voltage <= warn_voltage:
                            voltage_msg = "LOW BATT WARNING"
                        elif voltage <= min_voltage:
                            voltage_msg = "ULTRA LOW BATT!!!"
                        elif voltage >= max_voltage:
                            voltage_msg = "VOLTAGE TOO HIGH"

                        screen.addstr(8, 0, "Battery Voltage: {:2.2f}V".format(board.ANALOG['voltage']))
                        screen.clrtoeol()
                        screen.addstr(8, 24, voltage_msg, curses.A_BOLD + curses.A_BLINK)
                        screen.clrtoeol()

                    elif next_msg == 'MSP_STATUS_EX':
                        ARMED = board.bit_check(board.CONFIG['mode'],0)
                        screen.addstr(5, 0, "ARMED: {}".format(ARMED), curses.A_BOLD)
                        screen.clrtoeol()

                        screen.addstr(5, 50, "armingDisableFlags: {}".format(board.process_armingDisableFlags(board.CONFIG['armingDisableFlags'])))
                        screen.clrtoeol()

                        screen.addstr(6, 0, "cpuload: {}".format(board.CONFIG['cpuload']))
                        screen.clrtoeol()
                        screen.addstr(6, 50, "cycleTime: {}".format(board.CONFIG['cycleTime']))
                        screen.clrtoeol()

                        screen.addstr(7, 0, "mode: {}".format(board.CONFIG['mode']))
                        screen.clrtoeol()

                        screen.addstr(7, 50, "Flight Mode: {}".format(board.process_mode(board.CONFIG['mode'])))
                        screen.clrtoeol()


                    elif next_msg == 'MSP_MOTOR':
                        screen.addstr(9, 0, "Motor Values: {}".format(board.MOTOR_DATA))
                        screen.clrtoeol()

                    elif next_msg == 'MSP_RC':
                        screen.addstr(10, 0, "RC Channels Values: {}".format(board.RC['channels']))
                        screen.clrtoeol()

                    screen.addstr(11, 0, "GUI cycleTime: {0:2.2f}ms (average {1:2.2f}Hz)".format((last_cycleTime)*1000,
                                                                                                1/(sum(average_cycle)/len(average_cycle))))
                    screen.clrtoeol()

                    screen.addstr(3, 0, cursor_msg)
                    screen.clrtoeol()
                    

                end_time = time.time()
                last_cycleTime = end_time-start_time
                if (end_time-start_time)<CTRL_LOOP_TIME:
                    time.sleep(CTRL_LOOP_TIME-(end_time-start_time))
                    
                average_cycle.append(end_time-start_time)
                average_cycle.popleft()

    finally:
        screen.addstr(5, 0, "Disconneced from the FC!")
        screen.clrtoeol()


def connect_and_read_alt_from_FC():
    from yamspy import MSPy
    serial_port = SERIAL_PORT

    with MSPy(device=serial_port, loglevel='DEBUG', baudrate=115200) as board:
        if board.send_RAW_msg(MSPy.MSPCodes['MSP_ALTITUDE'], data=[]):
            dataHandler = board.receive_msg()
            board.process_recv_data(dataHandler)
            print(board.SENSOR_DATA['altitude'])

def connect_to_FC():
    from yamspy import MSPy
    serial_port = SERIAL_PORT

    #Need to reimpliment how we are accessing the FC if we want to return an object that allows us to send data

    # with MSPy(device=serial_port, loglevel='DEBUG', baudrate=115200) as board:
    #     if board.send_RAW_msg(MSPy.MSPCodes['MSP_ALTITUDE'], data=[]):
    #         dataHandler = board.receive_msg()
    #         board.process_recv_data(dataHandler)
    #         print(board.SENSOR_DATA['altitude'])


def send_cmd_to_FC(CMD):
    from yamspy import MSPy
    serial_port = SERIAL_PORT

    with MSPy(device=serial_port, loglevel='DEBUG', baudrate=115200) as board:
        if board.send_RAW_msg(MSPy.MSPCodes[CMD], data=[]):
            dataHandler = board.receive_msg()
            board.process_recv_data(dataHandler)
            # print(board.SENSOR_DATA['altitude'])




def main(opt):
    # check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

# This is commented because we want to run this more like a script rather than a program
if __name__ == '__main__':

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    
    #YOLO functions
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
        parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        # print_args(vars(opt))
        return opt

    def set_root():
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]  # YOLOv5 root directory
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


    def select_device(device='', batch_size=0, newline=True):
        # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
        s = f'YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
        device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
        cpu = device == 'cpu'
        mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
        if cpu or mps:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
            assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
                f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

        if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
            devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
            n = len(devices)  # device count
            if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
                assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
            space = ' ' * (len(s) + 1)
            for i, d in enumerate(devices):
                p = torch.cuda.get_device_properties(i)
                s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
            arg = 'cuda:0'
        elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
            s += 'MPS\n'
            arg = 'mps'
        else:  # revert to CPU
            s += 'CPU\n'
            arg = 'cpu'

        if not newline:
            s = s.rstrip()
        LOGGER.info(s)
        return torch.device(arg)



    def run(
            weights=ROOT / 'yolov5s.pt',  # model path or triton URL
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_csv=False,  # save results in CSV format
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=True,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        print("commented out this funciton")
        source = str(source)
        
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        webcam = source.isnumeric()
        screenshot = source.lower().startswith('screen')
        
        
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        # stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # # Run inference
        # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        # seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # for path, im, im0s, vid_cap, s in dataset:
        #     with dt[0]:
        #         im = torch.from_numpy(im).to(model.device)
        #         im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        #         im /= 255  # 0 - 255 to 0.0 - 1.0
        #         if len(im.shape) == 3:
        #             im = im[None]  # expand for batch dim

        #     # Inference
        #     with dt[1]:
        #         visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        #         pred = model(im, augment=augment, visualize=visualize)
        #         print(pred)

        #     # NMS
        #     with dt[2]:
        #         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        #     # Second-stage classifier (optional)
        #     # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        #     # Define the path for the CSV file
        #     csv_path = save_dir / 'predictions.csv'

        #     # Create or append to the CSV file
        #     def write_to_csv(image_name, prediction, confidence):
        #         data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
        #         with open(csv_path, mode='a', newline='') as f:
        #             writer = csv.DictWriter(f, fieldnames=data.keys())
        #             if not csv_path.is_file():
        #                 writer.writeheader()
        #             writer.writerow(data)

        #     # Process predictions
        #     for i, det in enumerate(pred):  # per image
        #         seen += 1
        #         if webcam:  # batch_size >= 1
        #             p, im0, frame = path[i], im0s[i].copy(), dataset.count
        #             s += f'{i}: '
        #         else:
        #             p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        #         p = Path(p)  # to Path
        #         save_path = str(save_dir / p.name)  # im.jpg
        #         txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        #         s += '%gx%g ' % im.shape[2:]  # print string
        #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #         imc = im0.copy() if save_crop else im0  # for save_crop
        #         annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        #         if len(det):
        #             # Rescale boxes from img_size to im0 size
        #             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        #             # Print results
        #             for c in det[:, 5].unique():
        #                 n = (det[:, 5] == c).sum()  # detections per class
        #                 s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        #             # Write results
        #             for *xyxy, conf, cls in reversed(det):
        #                 c = int(cls)  # integer class
        #                 label = names[c] if hide_conf else f'{names[c]}'
        #                 confidence = float(conf)
        #                 confidence_str = f'{confidence:.2f}'

        #                 if save_csv:
        #                     write_to_csv(p.name, label, confidence_str)

        #                 if save_txt:  # Write to file
        #                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #                     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        #                     with open(f'{txt_path}.txt', 'a') as f:
        #                         f.write(('%g ' * len(line)).rstrip() % line + '\n')

        #                 if save_img or save_crop or view_img:  # Add bbox to image
        #                     c = int(cls)  # integer class
        #                     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        #                     annotator.box_label(xyxy, label, color=colors(c, True))
        #                 if save_crop:
        #                     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        #         # Stream results
        #         im0 = annotator.result()
        #         if view_img:
        #             if platform.system() == 'Linux' and p not in windows:
        #                 windows.append(p)
        #                 cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        #                 cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        #             cv2.imshow(str(p), im0)
        #             cv2.waitKey(1)  # 1 millisecond

        #         # Save results (image with detections)
        #         if save_img:
        #             if dataset.mode == 'image':
        #                 cv2.imwrite(save_path, im0)
        #             else:  # 'video' or 'stream'
        #                 if vid_path[i] != save_path:  # new video
        #                     vid_path[i] = save_path
        #                     if isinstance(vid_writer[i], cv2.VideoWriter):
        #                         vid_writer[i].release()  # release previous video writer
        #                     if vid_cap:  # video
        #                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #                     else:  # stream
        #                         fps, w, h = 30, im0.shape[1], im0.shape[0]
        #                     save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        #                     vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #                 vid_writer[i].write(im0)

        #     # Print time (inference-only)
        #     LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # # Print results
        # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # if update:
        #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


    # set_root()
    opt = parse_opt()
    print(opt)
    main(opt)



#Testing simpleUI 
# run_curses(keyboard_controller)

#Testing that we can hit the FC
# read_alt_from_FC()

#Testing yolo model
# opt = parse_opt()
# main(opt)

