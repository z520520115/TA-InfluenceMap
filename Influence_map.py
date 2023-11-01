import os
import cv2
import numpy as np
import math
import glob

PATH = os.path.dirname(os.path.realpath(__file__))
folder_name = '\influence_map/'

accident_size = []


def get_video_frame(video_name):
    video_path = PATH + folder_name + '/{}.mp4'.format(video_name)
    video = cv2.VideoCapture(video_path)

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(height), int(width), 3)

    return size


# def distance_and_draw(img, COLOR, location):
#     width, height, _ = img.shape
#     CIRCLE_RATE = int(width * height / 10000)
#
#     loc_size = []
#     for main in location:
#         lengths = [math.dist(main, sub) for sub in location]
#         lengths.sort()
#         if len(lengths) >= 2:
#             size = 3 + int(3 * (CIRCLE_RATE / (lengths[1] + 10)))
#             if size > 10:
#                 size = 10
#             print(size)
#         else:
#             size = 1
#
#         loc_size.append([main, size])
#
#     for loc, size in loc_size:
#         new_img = np.zeros(img.shape, np.uint8)
#         loc = [int(loc[0]), int(loc[1])]
#         cv2.circle(new_img, loc, size, [COLOR, 0, 0], -1)
#         img = cv2.add(img, new_img)
#
#     return img

'''
函数的主要目的是根据给定的车辆位置信息，在图像上绘制表示车辆位置的圆形标记。输入参数为图像(img)，颜色(COLOR)，以及车辆位置列表(location)。该函数的实现细节如下：

从输入图像的形状中提取宽度、高度和通道数。接着计算CIRCLE_RATE（基于宽度和高度的平方和的平方根）和MIN_RATE（宽度和高度中的较小值）。

初始化一个空列表loc_size，用于存储车辆位置及其对应的圆形标记大小。

遍历location列表中的每个车辆位置（main）。对于每个车辆位置，计算它与其他所有位置（sub）的欧氏距离。然后，对距离列表进行排序，获取第二小的距离值。

根据第二小的距离值计算圆形标记的大小（size）。如果距离较小，将使用较大的圆形标记；否则，圆形标记将逐渐变小。将车辆位置及其对应的圆形标记大小添加到loc_size列表中。

遍历loc_size列表中的每个元素（车辆位置和对应的圆形标记大小）。对于每个元素，创建一个与输入图像形状相同的全零图像（new_img）。将车辆位置坐标转换为整数，并在new_img上绘制具有指定颜色、位置和大小的圆形标记。使用cv2.add()函数将new_img与输入图像相加。

将修改后的图像返回。

通过这个函数，可以根据车辆之间的距离在图像上绘制不同大小的圆形标记，以便更好地可视化车辆在场景中的分布情况。
'''

def distance_and_draw(img, COLOR, location):
    width, height, _ = img.shape
    CIRCLE_RATE = int(math.sqrt(pow(width, 2) + pow(height, 2)))
    MIN_RATE = min(width, height)
    FRAME_RATE = int(100 * width * height / (1280*720))

    loc_size = []
    for main in location:
        # lengths = [math.dist(main, sub) for sub in location]
        for sub in location:
            lengths = [math.sqrt(((int(main[0]) - int(sub[0])) ** 2) + ((int(main[1]) - int(sub[1])) ** 2) )]
            min_length = sorted(lengths)
            if len(min_length) >= 2:
                avg_len = int((min_length[1] / ((CIRCLE_RATE + MIN_RATE) / 2)) * 100)
                # 4 ~ 10
                if avg_len <= 10:
                    size = 15
                else:
                    count = (avg_len - 10) // 2
                    size = 15 - count
                    if size < 4:
                        size = 4
            else:
                size = 4

            loc_size.append([main, size])

    for loc, size in loc_size:
        new_img = np.zeros(img.shape, np.uint8)
        loc = [int(loc[0]), int(loc[1])]
        cv2.circle(new_img, loc, size, [COLOR, 0, 0], -1)
        img = cv2.add(img, new_img)

    return img

'''
函数的主要目的是根据给定的车辆尺寸信息，在图像上绘制表示车辆中心位置的圆形标记。输入参数为图像(img)和包含车辆尺寸数据的列表(data)。该函数的实现细节如下：

从输入图像的形状中提取宽度和高度。

遍历包含车辆尺寸数据的列表。对于每个尺寸元素，计算车辆的宽度、高度和中心位置。然后计算车辆尺寸相对于图像尺寸的百分比（car_search）。

创建一个与输入图像形状相同的全零图像（new_img）。

根据车辆尺寸相对于图像尺寸的百分比，判断圆形标记的大小。如果车辆尺寸大于阈值（car_search >= 5），在车辆中心位置绘制半径为30的圆形标记；否则，绘制半径为10的圆形标记。

使用cv2.add()函数将new_img与输入图像相加。

返回修改后的图像。

通过这个函数，可以根据车辆的尺寸在图像上绘制不同大小的圆形标记，以便更好地可视化车辆在场景中的大小关系。这有助于分析不同大小车辆在场景中的分布以及可能存在的碰撞风险。
'''


def car_size(img, data):
    im_width, im_height, _ = img.shape

    for size in data:
        width = abs(size[2] - size[3])
        height = abs(size[0] - size[1])
        center = [size[2] + width//2, size[0] + height//2]
        car_search = (width/im_width * 100 + height/im_height * 100)/2
        new_img = np.zeros(img.shape, np.uint8)
        # if car_search >= 5:
        #     cv2.rectangle(new_img, (size[2], size[0]), (size[3], size[1]), [0, 3, 0], -1)
        # else:
        #     cv2.rectangle(new_img, (size[2], size[0]), (size[3], size[1]), [0, 0, 255], -1)
        if car_search >= 5:
            cv2.circle(new_img, center, 30, [0, 3, 0], -1)
        else:
            cv2.circle(new_img, center, 10, [0, 3, 0], -1)
        img = cv2.add(img, new_img)

    return img

'''
函数的主要目的是根据给定的车辆轨迹数据，在图像上绘制车辆运动路线。输入参数为图像(img)，帧数(f_num)和包含车辆轨迹数据的字典(data)。该函数的实现细节如下：

遍历包含车辆轨迹数据的字典。对于字典中的每个键（key），初始化一个空列表(lines)，用于存储车辆轨迹中的位置。

遍历字典中键对应的值（轨迹数据），获取每个车辆在给定帧数之前的位置。将位置添加到lines列表中。

如果lines列表为空，则跳过当前循环，继续处理下一个键。

创建一个与输入图像形状相同的全零图像（new_img）。

如果lines列表包含至少两个位置，使用cv2.polylines()函数在new_img上绘制车辆轨迹（折线）。如果只包含一个位置，使用cv2.circle()函数在new_img上绘制一个圆形标记来表示该位置。

使用cv2.add()函数将new_img与输入图像相加。

返回修改后的图像。

通过这个函数，可以根据车辆的轨迹数据在图像上绘制车辆运动路线。这有助于分析车辆在场景中的移动模式，以便更好地了解车辆的行驶轨迹以及可能存在的碰撞风险。
'''


def car_route(img, f_num, data):
    for key in data.keys():
        lines = []
        for idx, loc in data[key]:
            if idx > f_num:
                break
            loc = [int(loc[0]), int(loc[1])]
            lines.append(loc)

        if len(lines) == 0:
            continue

        new_img = np.zeros(img.shape, np.uint8)
        if len(lines) >= 2:
            pts = np.array(lines, np.int32)
            cv2.polylines(new_img, [pts], False, [0, 0, 255], 4)
        else:
            cv2.circle(new_img, lines[0], 3, [0,0,255], -1)
        img = cv2.add(img, new_img)

    return img


def read_file():
    file_list = os.listdir(PATH + folder_name)
    file_list = set([name.split('.')[0] for name in file_list])
    if not os.path.isdir(PATH + '/influence_map/'):
        os.mkdir(PATH + '/influence_map/')

    for file_name in file_list:
        All_car = {}
        car_loc = {}
        V_size = get_video_frame(file_name)
        img = np.zeros(V_size, np.uint8)
        with open(PATH + folder_name + '/' + file_name + '.txt') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[-1] == '\n':
                    line = line[:-1]
                line = line[:-1]
                sp_line = line.split('[', 1)
                line = sp_line[0] + '[[' + sp_line[1] + ']}'
                line = eval(line)
                key = list(line.keys())[1]
                line_values = list(line.values())
                if line_values[0] not in All_car:
                    All_car[line_values[0]] = []
                All_car[line_values[0]].append(line_values[1])
                if key not in car_loc:
                    car_loc[key] = []
                car_loc[key].append([line_values[0], line_values[1][0]])

        for key in All_car.keys():
            location = [loc[0] for loc in All_car[key]]
            rect = [size[1] for size in All_car[key]]
            new_img = np.zeros(img.shape, np.uint8)
            COLOR = 4 + int(60 / line_values[0]) * key  # TODO: 그림 색 변경
            new_img = distance_and_draw(new_img, COLOR, location)
            img = cv2.add(img, new_img)

            new_img = np.zeros(img.shape, np.uint8)
            new_img = car_size(new_img, rect)
            img = cv2.add(img, new_img)

            new_img = np.zeros(img.shape, np.uint8)
            new_img = car_route(new_img, key, car_loc)
            img = cv2.add(img, new_img)

            path = PATH + '/influence_map/{}'.format(file_name)
            if not os.path.isdir(path):
                os.mkdir(path)
            cv2.imwrite(path + '/{}_{:05d}_4D_im.png'.format(file_name, key), img)


read_file()
print("Complete")
