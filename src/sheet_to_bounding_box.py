#!/usr/bin/env python

import numpy as np
import sys
import cv2
from pdf2image import convert_from_path
from threading import Thread

def image_resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the img to be resized and
    # grab the img size
    dim = None
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original img
    if width is None and height is None:
        return img

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the img
    resized = cv2.resize(img, dim, interpolation=inter)

    # return the resized image
    return resized


def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


def remove_lines(img):
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)
    # Show binary image
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    ratio = img.shape[0] / 2
    horizontal_size = cols / ratio
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontal_size), 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows / ratio
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(verticalsize)))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    # Inverse vertical image
    vertical = cv2.bitwise_not(vertical)
    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. img.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''
    # Step 1
    edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 3, -2)
    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel)
    # Step 3
    smooth = np.copy(vertical)
    # Step 4
    smooth = cv2.blur(smooth, (2, 2))
    # Step 5
    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]
    ret, thresh1 = cv2.threshold(vertical, 190, 255, cv2.THRESH_BINARY)
    thresh1 = cv2.erode(thresh1, kernel, iterations=1)
    return thresh1



class getBoundingBoxes(Thread):
    def __init__(self, img, begin, end):
        super().__init__()
        self.begin = begin
        self.end = end
        self.gave_infos = False
        self.finished = False
        self.boxes = []
        self.recurse = 0
        self.highest_recurse = 0
        self.lowest_yx = [0, 0]
        self.highest_yx = [0, 0]
        self.img = img.tolist()
        self.ndimg = img

    '''def __recursiveSearch(self, y, x):
        traceback.print_tf()
        self.recurse += 1
        if self.highest_recurse < self.recurse:
            self.highest_recurse = self.recurse
        if self.lowest_yx[0] > y:
            self.lowest_yx[0] = y
        if self.highest_yx[0] < y:
            self.highest_yx[0] = y
        if self.lowest_yx[1] > x:
            self.lowest_yx[1] = x
        if self.highest_yx[1] < x:
            self.highest_yx[1] = x
        self.img[y][x] = 190

        if y - 1 >= 0 and self.img[y - 1][x] == 0:
            self.__recursiveSearch(y - 1, x)
        if y + 1 < self.img.shape[0] and self.img[y + 1][x] == 0:
            self.__recursiveSearch(y + 1, x)
        if x - 1 >= 0 and self.img[y][x - 1] == 0:
            self.__recursiveSearch(y, x - 1)
        if x + 1 < self.img.shape[1] and self.img[y][x + 1] == 0:
            self.__recursiveSearch(y, x + 1)'''

    def get_move_value(self, ppos):
        if ppos[0] >= 0 and ppos[0] < len(self.img) and ppos[1] >= 0 and ppos[1] < len(self.img[0]):
            if self.img[ppos[0]][ppos[1]] == 0:
                return 1
        return 0

    def get_pos_copy(self, ppos):
        copy = []
        copy.append(ppos[0])
        copy.append(ppos[1])
        return copy

    def find_nodes_to_check(self, ppos):
        nodes_to_check = []
        tmp = self.get_pos_copy(ppos)
        tmp[0] -= 1
        if self.get_move_value(tmp) != 0:
            nodes_to_check.append(tmp)
        tmp = self.get_pos_copy(ppos)
        tmp[1] += 1
        if self.get_move_value(tmp) != 0:
            nodes_to_check.append(tmp)
        tmp = self.get_pos_copy(ppos)
        tmp[0] += 1
        if self.get_move_value(tmp) != 0:
            nodes_to_check.append(tmp)
        tmp = self.get_pos_copy(ppos)
        tmp[1] -= 1
        if self.get_move_value(tmp) != 0:
            nodes_to_check.append(tmp)
        return nodes_to_check

    def check_nodes_update_img(self, nodes):
        for node in nodes:
            if self.lowest_yx[0] > node[0]:
                self.lowest_yx[0] = node[0]
            if self.highest_yx[0] < node[0]:
                self.highest_yx[0] = node[0]
            if self.lowest_yx[1] > node[1]:
                self.lowest_yx[1] = node[1]
            if self.highest_yx[1] < node[1]:
                self.highest_yx[1] = node[1]
            self.img[node[0]][node[1]] = 190

    def find_new_nodes_to_check_from_old_ones(self, nodes):
        nodes_to_check = []
        for node in nodes:
            tmp = self.find_nodes_to_check(node)
            for to_append in tmp:
                if to_append in nodes_to_check:
                    continue
                nodes_to_check.append(to_append)
        return nodes_to_check

    def iterative_dijkstra(self, ppos):
        nodes = self.find_nodes_to_check(ppos)
        while 1:
            self.check_nodes_update_img(nodes)
            nodes = self.find_new_nodes_to_check_from_old_ones(nodes)
            if len(nodes) == 0:
                break

    def _get_bounding_box(self, y, x):
        self.lowest_yx = [y, x]
        self.highest_yx = [y, x]
        self.recurse = 0
        self.iterative_dijkstra([y, x])
        self.boxes.append((self.lowest_yx, self.highest_yx))
        cv2.rectangle(self.ndimg, (self.lowest_yx[1], self.lowest_yx[0]), (self.highest_yx[1], self.highest_yx[0]), (127, 127, 127), 1)

    def run(self):
        for y in range(self.begin, self.end):
            for x in range(len(self.img[y])):
                if self.img[y][x] == 0:
                    self._get_bounding_box(y, x)
        self.finished = True
        return self.img


def get_extremums(left_yx, right_yx,  left_yx2, right_yx2):
    left_x = left_yx[1] if left_yx[1] <= left_yx2[1] else left_yx2[1]
    left_y = left_yx[0] if left_yx[0] <= left_yx2[0] else left_yx2[0]
    right_x = right_yx[1] if right_yx[1] >= right_yx2[1] else right_yx2[1]
    right_y = right_yx[0] if right_yx[0] >= right_yx2[0] else right_yx2[0]
    return [(left_y, left_x), (right_y, right_x)]


def check_intersects(left_yx, right_yx,  left_yx2, right_yx2):
    #top right
    if (left_yx[1] <= left_yx2[1] <= right_yx[1]) and (left_yx[0] <= right_yx2[0] <= right_yx[0]):
        return True, get_extremums(left_yx, right_yx, left_yx2, right_yx2)
    #bottom right
    if (left_yx[1] <= left_yx2[1] <= right_yx[1]) and (left_yx[0] <= left_yx2[0] <= right_yx[0]):
        return True, get_extremums(left_yx, right_yx, left_yx2, right_yx2)
    #top left
    if (left_yx[1] <= right_yx2[1] <= right_yx[1]) and (left_yx[0] <= right_yx2[0] <= right_yx[0]):
        return True, get_extremums(left_yx, right_yx, left_yx2, right_yx2)
    #bottom left
    if (left_yx[1] <= right_yx2[1] <= right_yx[1]) and (left_yx[0] <= left_yx2[0] <= right_yx[0]):
        return True, get_extremums(left_yx, right_yx, left_yx2, right_yx2)
    return False, 0


def fusion_boxes(boxes):
    len_boxes = len(boxes)
    first = 0
    while first < len_boxes:
        second = first + 1
        while second < len_boxes:
            ret, new_box = check_intersects(boxes[first][0], boxes[first][1], boxes[second][0], boxes[second][1])
            if not ret:
                ret, new_box = check_intersects(boxes[second][0], boxes[second][1], boxes[first][0], boxes[first][1])
            if ret:
                boxes.pop(second)
                boxes.pop(first)
                boxes.append(new_box)
                len_boxes -= 1
                second = first
            second += 1
        first += 1
    return boxes


def launch_threaded_searchers(img, original, nb_thread):
    thread_list = []
    blocksize = img.shape[0] // nb_thread
    j = [0] * (nb_thread + 1)
    idx = 1
    for i in range(nb_thread):
        j[idx] = j[idx - 1]
        while True:
            black_pixel = 0
            for x in range(img.shape[1]):
                if (blocksize * i) + blocksize + j[idx] < img.shape[0] and img[(blocksize * i) + blocksize + j[idx]][
                    x] != 255:
                    black_pixel += 1
                    j[idx] += 1
            if black_pixel == 0:
                idx += 1
                break
        if i == nb_thread - 1 or ((blocksize * i) + blocksize + j[idx - 1]) + 1 >= img.shape[0]:
            boundingBoxes = getBoundingBoxes(img, (blocksize * i) + j[idx - 2], img.shape[0])
            boundingBoxes.setName(str(i))
            thread_list.append(boundingBoxes)
            break
        boundingBoxes = getBoundingBoxes(img, (blocksize * i) + j[idx - 2], (blocksize * i) + blocksize + j[idx - 1])
        boundingBoxes.setName(str(i))
        thread_list.append(boundingBoxes)

    for thread in thread_list:
        thread.start()

    exit = False
    finished = 0
    boxes = []
    while finished < idx - 1 or exit:
        for thread in thread_list:
            if thread.finished == True and thread.gave_infos == False:
                thread.gave_infos = True
                boxes += thread.boxes
                finished += 1
            if thread.getName() == '0':
                cv2.imshow(thread.getName(), image_resize(thread.ndimg, height=1080))
                if cv2.waitKey(1) & 0xFF == 27:
                    exit = True

    cv2.destroyAllWindows()
    boxes = fusion_boxes(boxes)
    for box in boxes:
        cv2.rectangle(original, (box[0][1], box[0][0]), (box[1][1], box[1][0]), (60, 60, 60), 1)

    for thread in thread_list:
        thread.keep_running = False
        thread.join()
    return original

def open_pdf(path, nb_threads=0):
    print('loading pdf...')
    images = convert_from_path(path,  300)
    it = 0
    for image in images:
        print('processing pdf page ' + str(it))
        img = np.array(image)
        process_image(img, nb_threads)
        it += 1
    print('finished process')


def process_image(img, nb_threads=0):
    img = image_resize(img, height=1080)
    original = img.copy()
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if img is None:
        print('Error opening image: ' + img)
        return -1

    img = remove_lines(img)
    img = launch_threaded_searchers(img, original, int(nb_threads))
    cv2.imshow('final', image_resize(img, height=1080))
    cv2.waitKey(0)

def main(argv):
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    nb_thread = 1 if len(argv) == 1 else int(argv[1])
    print(nb_thread)
    if argv[0][-4:] == '.pdf':
        open_pdf(argv[0], nb_thread)
    else:
        img = cv2.imread(argv[0], cv2.IMREAD_COLOR)
        process_image(img, nb_thread)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])