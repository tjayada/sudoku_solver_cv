import sys
import cv2
import numpy as np
import imutils

def black_white_seperated_sudoku(image):
    """"
    Returns a black/white floodfilled copy of the original image
    """
    try:
        im_in = image.copy()
    except:
        print("")
        print("")
        print("Error with image path. Check for mistakes")
        print("")
        print("")
        sys.exit()

    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]

    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    return im_floodfill_inv


def get_start_and_end_points(template, targets, start):
    """"
    Returns the exact location of the sudoku
    """
    diction = {}

    for target in targets:

        (tH, tW) = template.shape[:2]

        found = None

        for scale in np.linspace(0.2, 2.0, 40)[::-1]:

            resized = imutils.resize(target, width=int(target.shape[1] * scale))
            r = target.shape[1] / float(resized.shape[1])

            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        (maxVal, maxLoc, r) = found

        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        if start:
            diction[maxVal] = (startX, startY)
        else:
            diction[maxVal] = (endX, endY)

    return diction[max(diction)]


def load_templates():
    """"
    returns the templates for the numbers
    """
    im_1 = cv2.imread("templates/1.png", cv2.IMREAD_GRAYSCALE)
    im_2 = cv2.imread("templates/2.png", cv2.IMREAD_GRAYSCALE)
    im_3 = cv2.imread("templates/3.png", cv2.IMREAD_GRAYSCALE)
    im_4 = cv2.imread("templates/4.png", cv2.IMREAD_GRAYSCALE)
    im_5 = cv2.imread("templates/5.png", cv2.IMREAD_GRAYSCALE)
    im_6 = cv2.imread("templates/6.png", cv2.IMREAD_GRAYSCALE)
    im_7 = cv2.imread("templates/7.png", cv2.IMREAD_GRAYSCALE)
    im_8 = cv2.imread("templates/8.png", cv2.IMREAD_GRAYSCALE)
    im_9 = cv2.imread("templates/9.png", cv2.IMREAD_GRAYSCALE)

    all_numbers = [im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9]

    return all_numbers


def find_matches(target, template, all_points_of_interest, r):
    """"
    returns nested list containing rounded coordinates of number matches
    """
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.80
    loc = np.where(res >= threshold)

    points_of_interest = [(0, 0)]

    unique_match = True

    for pt in zip(*loc[::-1]):

        if unique_match:
            rounder1, rounder2 = int((pt[0] + w / 2) * r), int((pt[1] + h / 2) * r)
            rounder1, rounder2 = round(rounder1 / 10) * 10, round(rounder2 / 10) * 10
            points_of_interest.append((rounder1, rounder2))

        else:
            unique_match = True

    all_points_of_interest.append(points_of_interest)


def number_occurrences(list_of_numbers):
    """"
    divides nested list of coordinates into individual dictionaries containing occurrences
    """
    all_1 = []
    all_2 = []
    all_3 = []
    all_4 = []
    all_5 = []
    all_6 = []
    all_7 = []
    all_8 = []
    all_9 = []

    for elem in list_of_numbers:
        for idx, el in enumerate(elem):
            if idx == 0:
                all_1.append(el)
            elif idx == 1:
                all_2.append(el)
            elif idx == 2:
                all_3.append(el)
            elif idx == 3:
                all_4.append(el)
            elif idx == 4:
                all_5.append(el)
            elif idx == 5:
                all_6.append(el)
            elif idx == 6:
                all_7.append(el)
            elif idx == 7:
                all_8.append(el)
            elif idx == 8:
                all_9.append(el)

    all_1 = sum(all_1, [])
    unique_1 = set(all_1)

    all_2 = sum(all_2, [])
    unique_2 = set(all_2)

    all_3 = sum(all_3, [])
    unique_3 = set(all_3)

    all_4 = sum(all_4, [])
    unique_4 = set(all_4)

    all_5 = sum(all_5, [])
    unique_5 = set(all_5)

    all_6 = sum(all_6, [])
    unique_6 = set(all_6)

    all_7 = sum(all_7, [])
    unique_7 = set(all_7)

    all_8 = sum(all_8, [])
    unique_8 = set(all_8)

    all_9 = sum(all_9, [])
    unique_9 = set(all_9)

    #####################

    dic_1 = {}
    dic_2 = {}
    dic_3 = {}
    dic_4 = {}
    dic_5 = {}
    dic_6 = {}
    dic_7 = {}
    dic_8 = {}
    dic_9 = {}

    for uni in unique_1:
        dic_1[uni] = all_1.count(uni)

    for uni in unique_2:
        dic_2[uni] = all_2.count(uni)

    for uni in unique_3:
        dic_3[uni] = all_3.count(uni)

    for uni in unique_4:
        dic_4[uni] = all_4.count(uni)

    for uni in unique_5:
        dic_5[uni] = all_5.count(uni)

    for uni in unique_6:
        dic_6[uni] = all_6.count(uni)

    for uni in unique_7:
        dic_7[uni] = all_7.count(uni)

    for uni in unique_8:
        dic_8[uni] = all_8.count(uni)

    for uni in unique_9:
        dic_9[uni] = all_9.count(uni)

    return dic_1, dic_2, dic_3, dic_4, dic_5, dic_6, dic_7, dic_8, dic_9


def put_together_close_keys(dic):
    """"
    returns dictionary with combined equal coordinates that were rounded differently
    """
    if (0, 0) in list(dic.keys()):
        del dic[(0, 0)]

    for key in list(dic.keys()):
        if (key[0] - 10, key[1]) in list(dic.keys()):
            dic[key] += dic[(key[0] - 10, key[1])]
            dic.pop((key[0] - 10, key[1]), None)
            dic = put_together_close_keys(dic)
            return dic

        elif (key[0] - 10, key[1] - 10) in list(dic.keys()):
            dic[key] += dic[(key[0] - 10, key[1] - 10)]
            dic.pop((key[0] - 10, key[1] - 10), None)
            dic = put_together_close_keys(dic)
            return dic

        elif (key[0], key[1] - 10) in list(dic.keys()):
            dic[key] += dic[(key[0], key[1] - 10)]
            dic.pop((key[0], key[1] - 10), None)
            dic = put_together_close_keys(dic)
            return dic

        elif (key[0] - 10, key[1] + 10) in list(dic.keys()):
            dic[key] += dic[(key[0] - 10, key[1] + 10)]
            dic.pop((key[0] - 10, key[1] + 10), None)
            dic = put_together_close_keys(dic)
            return dic

        elif (key[0] + 10, key[1] - 10) in list(dic.keys()):
            dic[key] += dic[(key[0] + 10, key[1] - 10)]
            dic.pop((key[0] + 10, key[1] - 10), None)
            dic = put_together_close_keys(dic)
            return dic

        elif (key[0] + 10, key[1] + 10) in list(dic.keys()):
            dic[key] += dic[(key[0] + 10, key[1] + 10)]
            dic.pop((key[0] + 10, key[1] + 10), None)
            dic = put_together_close_keys(dic)
            return dic

        elif (key[0], key[1] + 10) in list(dic.keys()):
            dic[key] += dic[(key[0], key[1] + 10)]
            dic.pop((key[0], key[1] + 10), None)
            dic = put_together_close_keys(dic)
            return dic

        elif (key[0] + 10, key[1]) in list(dic.keys()):
            dic[key] += dic[(key[0] + 10, key[1])]
            dic.pop((key[0] + 10, key[1]), None)
            dic = put_together_close_keys(dic)
            return dic

    return dic


def determine_max(dic):
    """
    returns the max value (of occurrences) and respective coordinate of given dictionary
    """
    if len(dic) >= 1:
        return max(dic, key=dic.get), max(dic.values())
    else:
        return (0, 0)


def find_max(dic_1, dic_2, dic_3, dic_4, dic_5, dic_6, dic_7, dic_8, dic_9):
    """
    finds (max) values of all max values and returns coordinate and number
    """
    max_1 = determine_max(dic_1)
    max_2 = determine_max(dic_2)
    max_3 = determine_max(dic_3)
    max_4 = determine_max(dic_4)
    max_5 = determine_max(dic_5)
    max_6 = determine_max(dic_6)
    max_7 = determine_max(dic_7)
    max_8 = determine_max(dic_8)
    max_9 = determine_max(dic_9)

    maxes = [0, max_1, max_2, max_3, max_4, max_5, max_6, max_7, max_8, max_9]
    maxes_values = [0, max_1[1], max_2[1], max_3[1], max_4[1], max_5[1], max_6[1], max_7[1], max_8[1], max_9[1]]

    maximilian = max(maxes_values)

    idx_maximilian = maxes_values.index(maximilian)

    coordinates_maximilian = maxes[idx_maximilian][0]

    return coordinates_maximilian, idx_maximilian


def del_maximilian(dic, coord):
    """
    returns dictionary after (max) value got deleted
    """
    del dic[coord]

    return dic


def color_me_in(sudoku_array, x_steps, y_steps, coordinate, value):
    """
    maps coordinates into 9x9 array
    """
    for x in range(9):
        for y in range(9):

            if x_steps * x < coordinate[1] < x_steps * (x + 1):
                if y_steps * y < coordinate[0] < y_steps * (y + 1):
                    if sudoku_array[x, y] == 0:
                        sudoku_array[x, y] = value

    return sudoku_array


def print_grid(grid):
    """
    prints the sudoku in a more beautiful way
    """
    print()
    for i, elem in enumerate(grid):
        if i % 3 == 0:
            print()
        print(elem[0:3], elem[3:6], elem[6:9])
    print()
    print()


def possible(row, col, digit):
    """
    checks whether a certain "move" is legal or not
    """
    row_pos = row//3 * 3
    col_pos = col//3 * 3

    sub_grid_elements = grid[row_pos:row_pos+3, col_pos:col_pos+3]

    if digit in sub_grid_elements or digit in grid[row] or digit in grid[:, col]:
        return False
    else:
        return True


def solve():
    """
    solves the sudoku recursively
    """
    try:
        pos = np.argwhere(grid == 0)[0]
    except IndexError:
        return True

    for i in range(1, 10):
        if possible(pos[0], pos[1], i):
            grid[pos[0], pos[1]] = i
            if solve():
                return True
            else:
                grid[pos[0], pos[1]] = 0

    return False


if __name__ == '__main__':
    """
    combines all functions above to detect sudoku/numbers and print solution
    """
    global grid

    im = cv2.imread(str(sys.argv[1]), cv2.IMREAD_GRAYSCALE)

    new_image = black_white_seperated_sudoku(im)

    lt_template_for_array = cv2.imread("square_mapping/lt-basic-square-outline.jpg", cv2.IMREAD_GRAYSCALE)
    br_template_for_array = cv2.imread("square_mapping/br-basic-square-outline.jpg", cv2.IMREAD_GRAYSCALE)

    start_point = get_start_and_end_points(lt_template_for_array, [new_image], True)
    end_point = get_start_and_end_points(br_template_for_array, [new_image], False)

    x_steps, y_steps = int((end_point[1] - start_point[1]) / 9), int((end_point[0] - start_point[0]) / 9)

    ret, thresh1 = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    new_image = thresh1[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    all_numbers = load_templates()

    scaled_poit = []
    thresh1 = new_image.copy()
    killer = new_image.copy()
    for scale in np.linspace(0.5, 2.5, 150)[::-1]:

        resized = imutils.resize(thresh1, width=int(thresh1.shape[1] * scale))
        r = thresh1.shape[1] / float(resized.shape[1])

        all_points_of_interest = []
        for i in range(9):
            find_matches(resized, all_numbers[i], all_points_of_interest, r)

        scaled_poit.append(all_points_of_interest)

    scaled_poit_wo_zeros = []
    for scales in scaled_poit:
        counter = 0
        for occurences in scales:

            if len(occurences) == 1:
                counter += 1

        if counter != 9:
            scaled_poit_wo_zeros.append(scales)

    dic_1, dic_2, dic_3, dic_4, dic_5, dic_6, dic_7, dic_8, dic_9 = number_occurrences(scaled_poit_wo_zeros)

    dic_1 = put_together_close_keys(dic_1)
    dic_2 = put_together_close_keys(dic_2)
    dic_3 = put_together_close_keys(dic_3)
    dic_4 = put_together_close_keys(dic_4)
    dic_5 = put_together_close_keys(dic_5)
    dic_6 = put_together_close_keys(dic_6)
    dic_7 = put_together_close_keys(dic_7)
    dic_8 = put_together_close_keys(dic_8)
    dic_9 = put_together_close_keys(dic_9)

    terminator = 1
    sudu = np.zeros((9, 9), dtype=int)

    while terminator > 0:
        coord_psi, valve = find_max(dic_1, dic_2, dic_3, dic_4, dic_5, dic_6, dic_7, dic_8, dic_9)

        if valve == 1:
            dic_1 = del_maximilian(dic_1, coord_psi)

        elif valve == 2:
            dic_2 = del_maximilian(dic_2, coord_psi)

        elif valve == 3:
            dic_3 = del_maximilian(dic_3, coord_psi)

        elif valve == 4:
            dic_4 = del_maximilian(dic_4, coord_psi)

        elif valve == 5:
            dic_5 = del_maximilian(dic_5, coord_psi)

        elif valve == 6:
            dic_6 = del_maximilian(dic_6, coord_psi)

        elif valve == 7:
            dic_7 = del_maximilian(dic_7, coord_psi)

        elif valve == 8:
            dic_8 = del_maximilian(dic_8, coord_psi)

        elif valve == 9:
            dic_9 = del_maximilian(dic_9, coord_psi)

        sudu = color_me_in(sudu, x_steps, y_steps, coord_psi, valve)

        terminator = len(dic_1) + len(dic_2) + len(dic_3) + len(dic_4) + len(dic_5) + len(dic_6) + len(dic_7) + len(
            dic_8) + len(dic_9)

    grid = sudu
    solve()
    print_grid(grid)
