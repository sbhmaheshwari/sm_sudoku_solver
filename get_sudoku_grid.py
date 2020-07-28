import cv2
import numpy as np
import operator
#import matplotlib.pyplot as plt

def distance_between(p1, p2):
  return(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

def preprocess_image(img, skip_dilate=False):
  # blur -> threshold -> dilate
  proc = cv2.GaussianBlur(img.copy(), (9,9), 0)
  proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  proc = cv2.bitwise_not(proc, proc)

  if not skip_dilate:
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)
  return(proc)

def get_corners(img):
  ext_contours, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  sortedContours = sorted(ext_contours, key = cv2.contourArea, reverse=True)
  polygon = sortedContours[0]
  bottom_right, _ = max(enumerate([pt[0][0]+pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
  top_right, _ = max(enumerate([pt[0][0]-pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
  bottom_left, _ = min(enumerate([pt[0][0]-pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
  top_left, _ = min(enumerate([pt[0][0]+pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
  req_points = [polygon[bottom_right][0], polygon[bottom_left][0], polygon[top_left][0], polygon[top_right][0]]
  return(req_points)

def get_warped(img, req_points):
  bottom_right, bottom_left, top_left, top_right = req_points
  src = np.array([bottom_right, bottom_left, top_left, top_right], dtype='float32')
  side = max([distance_between(bottom_right, bottom_left),
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(top_right, top_left)])
  dst = np.array([[side-1, side-1], [0, side - 1], [0, 0], [side - 1, 0]], dtype='float32')
  m = cv2.getPerspectiveTransform(src, dst)
  warped_image = cv2.warpPerspective(img, m, (int(side), int(side)))
  return(warped_image)

def infer_grid(img):
  squares = []
  side = img.shape[0]/9
  for j in range(9):
    for i in range(9):
      p1 = (i*side, j*side)
      p2 = ((i+1)*side, (j+1)*side)
      squares.append((p1, p2))
  return(squares)

def cut_from_rect(img, rect):
	"""Cuts a rectangle from an image using the top left and bottom right points."""
	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def get_digits(warped_img, squares, size = 28):
  digits = []
  for square in squares:
    rect = cut_from_rect(warped_img.copy(), square)
    digits.append(extract_features(rect, size = size))
  return(digits)

def extract_features(rect, size = None):
  height, width = rect.shape
  margin = int((height+width)/2/2.5)
  topl = [margin, margin]
  bottomr = [width-margin, height-margin]
  bbox = find_largest_feature(rect, topl, bottomr)
  imp_digit = cut_from_rect(rect, bbox)
  
  w = bbox[1][0] - bbox[0][0]
  h = bbox[1][1] - bbox[0][1]
  if w>0 and h>0 and w*h>100 and len(imp_digit)>0:
    imp_digit_pad = get_pad(imp_digit, size = size, margin = 4)
  else:
    imp_digit_pad = np.zeros((size, size), np.uint8)
  return(imp_digit_pad)

def get_pad(imp_digit, size = 28, margin = 0, background = 0):
  height, width = imp_digit.shape
  
  def center_pad(length):
    side1 = int((size-length)/2)
    side2 = side1 + length%2
    return(side1, side2)
  
  if height>width:
    t_pad = int(margin/2)
    b_pad = t_pad
    ratio = (size-margin)/height
    height, width = int(ratio*height), int(ratio*width)
    l_pad, r_pad = center_pad(width)
  
  else:
    l_pad = int(margin/2)
    r_pad = l_pad
    ratio = (size-margin)/width
    height, width = int(ratio*height), int(ratio*width)
    t_pad, b_pad = center_pad(height)

  imp_digit = cv2.resize(imp_digit, (width, height))
  imp_digit = cv2.copyMakeBorder(imp_digit, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
  imp_digit = cv2.resize(imp_digit, (size, size))
  return(imp_digit)

def find_largest_feature(rect, topl = None, bottomr = None):
  height, width = rect.shape
  img = rect.copy()
  if topl is None:
    topl = [0, 0]
  if bottomr is None:
    bottomr = [width, height]
  max_area = 0
  seed = (None, None)
  for x in range(topl[0], bottomr[0]):
    for y in range(topl[1], bottomr[1]):
      if img.item(y,x) == 255 and x < width and y < height:
        area = cv2.floodFill(img, None, (x,y), 64)
        if area[0]>max_area:
          seed = (x,y)
          max_area = area[0]
  
  for x in range(width):
    for y in range(height):
      if img.item(y,x)==255:
        cv2.floodFill(img, None, (x,y), 64)
  
  if all([p is not None for p in seed]):
    cv2.floodFill(img, None, seed, 255)

  top, left, bottom, right = height, width, 0, 0

  for x in range(width):
    for y in range(height):
      if img.item(y,x)==64:
        cv2.floodFill(img, None, (x,y), 0)
      if img.item(y,x)==255:
        top = y if top>y else top
        left = x if left>x else left
        bottom = y if bottom<y else bottom
        right = x if right<x else right

  bbox = [[left, top], [right, bottom]]
  return(np.array(bbox, dtype = 'float32'))
  
def get_sudoku(img = [], img_path = None):
  
  img_r = None
  if img_path is not None:
    img_r = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  elif len(img)>0:
    img_r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  else:
    print('path invalid')

  if len(img_r)>0:
    processed_img = preprocess_image(img_r)
    req_points = get_corners(processed_img)
    warped_image = get_warped(img_r, req_points)
    squares = infer_grid(warped_image)
    warped_processed = preprocess_image(warped_image.copy(), skip_dilate=True)
    digits = get_digits(warped_processed, squares)

  return(digits)

def plot_many_images(images, titles, rows=1, columns=2):
  plt.figure(figsize=(12,12))
  for i, image in enumerate(images):
    plt.subplot(rows, columns, i+1)
    plt.imshow(image, 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])  # Hide tick marks
  plt.show()
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
	"""Draws circular points on an image."""
	img = in_img.copy()

	# Dynamically change to a colour image if necessary
	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	for point in points:
		img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
	plt.imshow(img, 'gray')
	return img
def display_rects(in_img, rects, colour=(0, 0, 255)):
  img = in_img.copy()
  if len(colour) == 3:
    if len(img.shape) == 2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  for rect in rects:
    img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
  plt.imshow(img)
  return img
def show_digits(digits, colour=255):
	"""Shows list of 81 extracted digits in a grid format"""
	rows = []
	with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
	for i in range(9):
		row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis = 1)
		rows.append(row)
	plt.imshow(np.concatenate(rows), 'gray')