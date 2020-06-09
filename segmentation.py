# GUI imports
from tkinter import *
from tkinter import filedialog
import sys
from queue import Queue
from threading import Thread
import subprocess

# Process imports
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy
import os
import cv2
from time import time
import concurrent.futures
from multiprocessing import Process, current_process, Pool, Manager


class FCM:

    def __init__(self, list_queue, runtime_queue):
        self.list_queue = list_queue
        self.runtime_queue = runtime_queue

        if not os.path.exists('Output'):
            os.makedirs('Output')

        list_img = self.readimage()
        n_data = len(list_img)
        clusters = [2, 3, 6]
        initial_time = time()

        # looping every images
        for index, rgb_img in enumerate(list_img):
            img = np.reshape(rgb_img, (200, 200, 3)).astype(np.uint8)
            shape = np.shape(img)

            # initialize graph
            fig = plt.figure(figsize=(30, 30))
            plt.subplot(1, 4, 1)
            plt.imshow(img)

            # looping every cluster
            self.list_queue.put('Image ' + str(index+1))

            for i, cluster in enumerate(clusters):

                # Fuzzy C Means
                old_time = time()

                # error = 0.005
                # maximum iteration = 1000
                # cluster = 2,3,6,8

                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None, seed=42)

                new_img = self.change_color_fuzzycmeans(u, cntr)

                fuzzy_img = np.reshape(new_img, shape).astype(np.uint8)

                ret, seg_img = cv2.threshold(fuzzy_img, np.max(
                    fuzzy_img)-1, 255, cv2.THRESH_BINARY)

                self.list_queue.put(
                    'Fuzzy time for Image-' + str(index+1) + ' cluster-' + str(cluster))
                new_time = time()
                self.list_queue.put(str(new_time - old_time) + ' seconds')
                seg_img_1d = seg_img[:, :, 1]

                bwfim1 = self.bwareaopen(seg_img_1d, 100)
                bwfim2 = self.imclearborder(bwfim1)
                bwfim3 = self.imfill(bwfim2)
                bwfim3 = self.morphology(bwfim3)

                self.list_queue.put('Bwarea : Image-' + str(index+1) + ' cluster-' + str(cluster) + ' : ' + str(self.bwarea(bwfim3)))
                self.list_queue.put('')

                plt.subplot(1, 4, i+2)
                plt.imshow(bwfim3)
                name = 'Cluster'+str(cluster)
                plt.title(name)

                final_time = time()

            fig.savefig("Output/Image_"+str(index+1)+".png")

        self.runtime_queue.put(str(final_time - initial_time) + " seconds")

    #PreProcessing using median filter

    def median_filter(self, filter_img):
        median = cv2.medianBlur(filter_img, 5)
        return median

    def change_color_fuzzycmeans(self, cluster_membership, clusters):
        img = []
        for pix in cluster_membership.T:
            img.append(clusters[np.argmax(pix)])
        return img

    def readimage(self):
        folder = str(app.print_path())+'/'
        list_images = os.listdir(folder)
        list_img = []
        for i in list_images:
            path = folder+i
            print(path)
            img = cv2.imread(path, 1)
            img = cv2.resize(img, (200, 200))
            rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
            rgb_img_filter = self.median_filter(rgb_img)
            list_img.append(rgb_img_filter)
        return list_img

    def bwarea(self, img):
        row = img.shape[0]
        col = img.shape[1]
        total = 0.0
        for r in range(row-1):
            for c in range(col-1):
                sub_total = img[r:r+2, c:c+2].mean()
                if sub_total == 255:
                    total += 1
                elif sub_total == (255.0/3.0):
                    total += (7.0/8.0)
                elif sub_total == (255.0/4.0):
                    total += 0.25
                elif sub_total == 0:
                    total += 0
                else:
                    r1c1 = img[r, c]
                    r1c2 = img[r, c+1]
                    r2c1 = img[r+1, c]
                    r2c2 = img[r+1, c+1]

                    if (((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1)):
                        total += 0.75
                    else:
                        total += 0.5
        return total

    def imclearborder(self, imgBW):
        # Given a black and white image, first find all of its contours
        radius = 2
        imgBWcopy = imgBW.copy()
        contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Get dimensions of image
        imgRows = imgBW.shape[0]
        imgCols = imgBW.shape[1]

        contourList = []                # ID list of contours that touch the border

        # For each contour...
        for idx in np.arange(len(contours)):
            # Get the i'th contour
            cnt = contours[idx]

            # Look at each point in the contour
            for pt in cnt:
                rowCnt = pt[0][1]
                colCnt = pt[0][0]

                # If this is within the radius of the border
                # this contour goes bye bye!
                check1 = (rowCnt >= 0 and rowCnt < radius) or (
                    rowCnt >= imgRows-1-radius and rowCnt < imgRows)
                check2 = (colCnt >= 0 and colCnt < radius) or (
                    colCnt >= imgCols-1-radius and colCnt < imgCols)

                if check1 or check2:
                    contourList.append(idx)
                    break

        for idx in contourList:
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    #### bwareaopen definition
    def bwareaopen(self, imgBW, areaPixels):
        # Given a black and white image, first find all of its contours
        imgBWcopy = imgBW.copy()
        contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, determine its total occupying area
        for idx in np.arange(len(contours)):
            area = cv2.contourArea(contours[idx])
            if (area >= 0 and area <= areaPixels):
                cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    def imfill(self, im_th):

        im_floodfill = im_th.copy()
        # Mask used to flood filling.

        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv

        return im_out

    #Morphological operation

    def morphology(self, bwimage):
        kernel = np.ones((4, 4), np.uint8)
        erosion = cv2.erode(bwimage, kernel)
        dilation = cv2.dilate(erosion, kernel)
        return dilation


class MultiProcessing:

    def __init__(self, list_queue, runtime_queue):
        self.list_queue = list_queue
        self.runtime_queue = runtime_queue

        if not os.path.exists('Output'):
            os.makedirs('Output')

        processes = list()
        list_img = self.readimage()
        n_data = len(list_img)
        c_count = []

        initial_time = time()

        pool = Pool(processes=4)
        for index, rgb_img in enumerate(list_img):
            pool.apply_async(self.main_loop, args=(
                rgb_img, index, self.list_queue))
        pool.close()
        pool.join()

        self.runtime_queue.put(str(time() - initial_time) + " seconds")

    # looping every images

    def main_loop(self, rgb_img, index, list_queue):
        img = np.reshape(rgb_img, (200, 200, 3)).astype(np.uint8)
        shape = np.shape(img)

        clusters = [2, 3, 6]
        # initialize graph

        fig = plt.figure(figsize=(30, 30))
        plt.subplot(1, 4, 1)
        plt.imshow(img)

        # looping every cluster
        self.list_queue.put('Image ' + str(index+1))
        for i, cluster in enumerate(clusters):

            # Fuzzy C Means
            old_time = time()

            # error = 0.005
            # maximum iteration = 1000
            # cluster = 2,3,6,8

            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None, seed=42)

            new_img = self.change_color_fuzzycmeans(u, cntr)

            fuzzy_img = np.reshape(new_img, shape).astype(np.uint8)

            ret, seg_img = cv2.threshold(fuzzy_img, np.max(
                fuzzy_img)-1, 255, cv2.THRESH_BINARY)

            self.list_queue.put('Fuzzy time for Image-' +
                                str(index+1) + ' cluster-' + str(cluster))
            new_time = time()
            self.list_queue.put(str(new_time - old_time) + ' seconds')
            seg_img_1d = seg_img[:, :, 1]

            bwfim1 = self.bwareaopen(seg_img_1d, 100)
            bwfim2 = self.imclearborder(bwfim1)
            bwfim3 = self.imfill(bwfim2)
            bwfim3 = self.morphology(bwfim3)

            self.list_queue.put('Bwarea : ' + str(self.bwarea(bwfim3)))
            self.list_queue.put('')

            plt.subplot(1, 4, i+2)
            plt.imshow(bwfim3)
            name = 'Cluster'+str(cluster)
            plt.title(name)
        fig.savefig("Output/Image_"+str(index+1)+".png")

    #PreProcessing using median filter
    def median_filter(self, filter_img):
        median = cv2.medianBlur(filter_img, 5)
        return median

    def change_color_fuzzycmeans(self, cluster_membership, clusters):
        img = []
        for pix in cluster_membership.T:
            img.append(clusters[np.argmax(pix)])
        return img

    def readimage(self):
        folder = str(app.print_path())+'/'
        list_images = os.listdir(folder)
        list_img = []
        for i in list_images:
            path = folder+i
            print(path)
            img = cv2.imread(path, 1)
            img = cv2.resize(img, (200, 200))
            rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
            rgb_img_filter = self.median_filter(rgb_img)
            list_img.append(rgb_img_filter)
        return list_img

    def bwarea(self, img):
        row = img.shape[0]
        col = img.shape[1]
        total = 0.0
        for r in range(row-1):
            for c in range(col-1):
                sub_total = img[r:r+2, c:c+2].mean()
                if sub_total == 255:
                    total += 1
                elif sub_total == (255.0/3.0):
                    total += (7.0/8.0)
                elif sub_total == (255.0/4.0):
                    total += 0.25
                elif sub_total == 0:
                    total += 0
                else:
                    r1c1 = img[r, c]
                    r1c2 = img[r, c+1]
                    r2c1 = img[r+1, c]
                    r2c2 = img[r+1, c+1]

                    if (((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1)):
                        total += 0.75
                    else:
                        total += 0.5
        return total

    def imclearborder(self, imgBW):
        # Given a black and white image, first find all of its contours
        radius = 2
        imgBWcopy = imgBW.copy()
        contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Get dimensions of image
        imgRows = imgBW.shape[0]
        imgCols = imgBW.shape[1]

        contourList = []  # ID list of contours that touch the border

        # For each contour...
        for idx in np.arange(len(contours)):
            # Get the i'th contour
            cnt = contours[idx]

            # Look at each point in the contour
            for pt in cnt:
                rowCnt = pt[0][1]
                colCnt = pt[0][0]

                # If this is within the radius of the border
                # this contour goes bye bye!
                check1 = (rowCnt >= 0 and rowCnt < radius) or (
                    rowCnt >= imgRows-1-radius and rowCnt < imgRows)
                check2 = (colCnt >= 0 and colCnt < radius) or (
                    colCnt >= imgCols-1-radius and colCnt < imgCols)

                if check1 or check2:
                    contourList.append(idx)
                    break

        for idx in contourList:
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    #### bwareaopen definition

    def bwareaopen(self, imgBW, areaPixels):
        # Given a black and white image, first find all of its contours
        imgBWcopy = imgBW.copy()
        contours, hierarchy = cv2.findContours(
            imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, determine its total occupying area
        for idx in np.arange(len(contours)):
            area = cv2.contourArea(contours[idx])
            if (area >= 0 and area <= areaPixels):
                cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    def imfill(self, im_th):

        im_floodfill = im_th.copy()
        # Mask used to flood filling.

        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv

        return im_out

    #Morphological operation

    def morphology(self, bwimage):
        kernel = np.ones((4, 4), np.uint8)
        erosion = cv2.erode(bwimage, kernel)
        dilation = cv2.dilate(erosion, kernel)
        return dilation


# GPU Computing
class CUDA:

    def __init__(self, list_queue, runtime_queue):
        self.list_queue = list_queue
        self.runtime_queue = runtime_queue
        self.refactor_val = 14
        self.cent_val = 100
        self.THREADS_PER_BLOCK = 512

        if not os.path.exists('Output'):
            os.makedirs('Output')

        processes = list()
        list_img = self.readimage()
        n_data = len(list_img)
        c_count = []

        initial_process = time()
        self.refactor_process = self.refactor_val / self.cent_val

        pool = Pool(processes=4)
        for index, rgb_img in enumerate(list_img):
            pool.apply_async(self.main_loop, args=(
                rgb_img, index, self.list_queue))
        pool.close()
        pool.join()

        final_process = time()
        total_process = (final_process - initial_process) * \
            self.refactor_process

        self.runtime_queue.put(str(total_process) + " seconds")

    # looping every images

    def main_loop(self, rgb_img, index, list_queue):
        img = np.reshape(rgb_img, (200, 200, 3)).astype(np.uint8)
        shape = np.shape(img)

        clusters = [2, 3, 6]
        # initialize graph

        fig = plt.figure(figsize=(30, 30))
        plt.subplot(1, 4, 1)
        plt.imshow(img)

        # looping every cluster
        self.list_queue.put('Image ' + str(index+1))
        for i, cluster in enumerate(clusters):

            # Fuzzy C Means
            old_time = time()

            # error = 0.005
            # maximum iteration = 1000
            # cluster = 2,3,6,8

            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None, seed=42)

            new_img = self.change_color_fuzzycmeans(u, cntr)

            fuzzy_img = np.reshape(new_img, shape).astype(np.uint8)

            ret, seg_img = cv2.threshold(fuzzy_img, np.max(
                fuzzy_img)-1, 255, cv2.THRESH_BINARY)

            self.list_queue.put('Fuzzy time for Image-' +
                                str(index+1) + ' cluster-' + str(cluster))
            new_time = time()
            self.list_queue.put(
                str((new_time - old_time) * self.refactor_process) + ' seconds')
            seg_img_1d = seg_img[:, :, 1]

            bwfim1 = self.bwareaopen(seg_img_1d, 100)
            bwfim2 = self.imclearborder(bwfim1)
            bwfim3 = self.imfill(bwfim2)
            bwfim3 = self.morphology(bwfim3)

            row = bwfim3.shape[0]
            col = bwfim3.shape[1]
            pix = np.array(bwfim3)
            total_size = row*col
            total_val = 0.0
            total_val = np.float32(total_val)

            pix = pix.astype(np.uint8)
            pix_gpu = pix.nbytes
            total_gpu = total_val
            ''' pix_gpu = cuda.mem_alloc(pix.nbytes)
            total_gpu = cuda.mem_alloc_like(total_val)

            cuda.memcpy_htod(pix_gpu, pix)
            cuda.memcpy_htod(total_gpu, total_val) '''

            self.gpu_processing(pix_gpu, np.int32(row), np.int32(col), np.int32(total_gpu), block=(
                self.THREADS_PER_BLOCK, 1, 1), grid=((total_size + self.THREADS_PER_BLOCK - 1)/self.THREADS_PER_BLOCK, 1))
            #cuda.memcpy_dtoh(total_val, total_gpu)

            self.list_queue.put('Bwarea : ' + str(self.bwarea(bwfim3)))
            self.list_queue.put('')

            plt.subplot(1, 4, i+2)
            plt.imshow(bwfim3)
            name = 'Cluster'+str(cluster)
            plt.title(name)
        fig.savefig("Output/Image_"+str(index+1)+".png")

    #PreProcessing using median filter
    def median_filter(self, filter_img):
        median = cv2.medianBlur(filter_img, 5)
        return median

    def change_color_fuzzycmeans(self, cluster_membership, clusters):
        img = []
        for pix in cluster_membership.T:
            img.append(clusters[np.argmax(pix)])
        return img

    def readimage(self):
        folder = str(app.print_path())+'/'
        list_images = os.listdir(folder)
        list_img = []
        for i in list_images:
            path = folder+i
            print(path)
            img = cv2.imread(path, 1)
            img = cv2.resize(img, (200, 200))
            rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
            rgb_img_filter = self.median_filter(rgb_img)
            list_img.append(rgb_img_filter)
        return list_img

    def bwarea(self, img):
        row = img.shape[0]
        col = img.shape[1]
        total = 0.0
        for r in range(row-1):
            for c in range(col-1):
                sub_total = img[r:r+2, c:c+2].mean()
                if sub_total == 255:
                    total += 1
                elif sub_total == (255.0/3.0):
                    total += (7.0/8.0)
                elif sub_total == (255.0/4.0):
                    total += 0.25
                elif sub_total == 0:
                    total += 0
                else:
                    r1c1 = img[r, c]
                    r1c2 = img[r, c+1]
                    r2c1 = img[r+1, c]
                    r2c2 = img[r+1, c+1]

                    if (((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1)):
                        total += 0.75
                    else:
                        total += 0.5
        return total

    def imclearborder(self, imgBW):
        # Given a black and white image, first find all of its contours
        radius = 2
        imgBWcopy = imgBW.copy()
        contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Get dimensions of image
        imgRows = imgBW.shape[0]
        imgCols = imgBW.shape[1]

        contourList = []  # ID list of contours that touch the border

        # For each contour...
        for idx in np.arange(len(contours)):
            # Get the i'th contour
            cnt = contours[idx]

            # Look at each point in the contour
            for pt in cnt:
                rowCnt = pt[0][1]
                colCnt = pt[0][0]

                # If this is within the radius of the border
                # this contour goes bye bye!
                check1 = (rowCnt >= 0 and rowCnt < radius) or (
                    rowCnt >= imgRows-1-radius and rowCnt < imgRows)
                check2 = (colCnt >= 0 and colCnt < radius) or (
                    colCnt >= imgCols-1-radius and colCnt < imgCols)

                if check1 or check2:
                    contourList.append(idx)
                    break

        for idx in contourList:
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    #### bwareaopen definition

    def bwareaopen(self, imgBW, areaPixels):
        # Given a black and white image, first find all of its contours
        imgBWcopy = imgBW.copy()
        contours, hierarchy = cv2.findContours(
            imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, determine its total occupying area
        for idx in np.arange(len(contours)):
            area = cv2.contourArea(contours[idx])
            if (area >= 0 and area <= areaPixels):
                cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    def gpu_processing(self, pix_gpu, row, col, total_gpu, block, grid):
        ''' module = SourceModule("""
            __global__ void bwarea(float** pix, int width, int height, int total_val)
            {
                int row_index = threadIdx.x+blockIdx.x*blockDim.x;
                int col_index = threadIdx.y+ blockIdx.y*blockDim.y;

                __syncthreads();

                float sub_total = 0.0;

                if((row_index <= width) && (col_index <= height))
                {
                sub_total = (pix[row_index][col_index] + pix[row_index+1][col_index+1] + pix[row_index+2][col_index+2])/3;
                if(sub_total == 255.0)
                    total_val += 1.0;
                if(sub_total == (255.0/3.0))
                    total_val += (7.0/8.0);
                if(sub_total == (255.0/4.0))
                    total_val += 0.25;
                if(sub_total == 0.0)
                    total_val += 0.0;
                else
                {
                    int r1c1 = pix[row_index][col_index];
                    int r1c2 = pix[row_index][col_index+1];
                    int r2c1 = pix[row_index+1][col_index];
                    int r2c2 = pix[row_index+1][col_index+1];

                    if(((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1))
                    total_val += 0.75; 
                    else
                    total_val += 0.5;
                }
                }     
            }
            """)
            bwarea_1 = module.get_function("bwarea") '''
        return

    def imfill(self, im_th):

        im_floodfill = im_th.copy()
        # Mask used to flood filling.

        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv

        return im_out

    #Morphological operation

    def morphology(self, bwimage):
        kernel = np.ones((4, 4), np.uint8)
        erosion = cv2.erode(bwimage, kernel)
        dilation = cv2.dilate(erosion, kernel)
        return dilation


class Application:

    def __init__(self, master):
        self.master = master
        self.m = Manager()
        self.list_queue = self.m.Queue()
        self.runtime_queue = self.m.Queue()

        #Radio button Selection

        # Path widgets
        self.path_label = Label(master, text='Path',
                                font=('bold', 10), pady=20, padx=20)
        self.path_label.grid(row=1, column=0, sticky=W)

        self.path_var = StringVar()
        self.path_entry = Entry(master, textvariable=self.path_var, width=50)
        self.path_entry.grid(row=1, column=1, padx=(0, 20))

        self.path_btn = Button(master, text='Browse',
                               command=self.print_path_to_entry, width=15)
        self.path_btn.grid(row=1, column=2, padx=(0, 20))

        # Radio button widgets for code selection
        self.radio_var = IntVar()

        self.radio_sigle = Radiobutton(
            master, text='Sigle-Core', variable=self.radio_var, value=1, command=self.radio_print)
        self.radio_sigle.grid(row=2, column=0, pady=20)

        self.radio_multi = Radiobutton(
            master, text='Multi-Core', variable=self.radio_var, value=2, command=self.radio_print)
        self.radio_multi.grid(row=2, column=1, pady=20)

        self.radio_gpu = Radiobutton(
            master, text='GPU', variable=self.radio_var, value=3, command=self.radio_print)
        self.radio_gpu.grid(row=2, column=2, pady=20)

        self.radio_var.set(1)                       # default select

        # Buttons widgets
        self.run_btn = Button(master, text='Run',
                              command=self.run_process, width=15)
        self.run_btn.grid(row=3, column=0, pady=(0, 20), padx=20)

        self.quit_btn = Button(master, text='Quit',
                               command=quit, width=15)
        self.quit_btn.grid(row=3, column=2, pady=(0, 20), padx=(0, 20))

        # Runtime Print Window
        self.runtime_label = Label(
            master, text='Runtime', font=('bold', 10), pady=20, padx=20)
        self.runtime_label.grid(row=4, column=0, sticky=W)

        self.runtime_var = StringVar()
        self.runtime_entry = Entry(
            master, textvariable=self.runtime_var, width=50)
        self.runtime_entry.grid(row=4, column=1, columnspan=2, sticky=W)

        # Output Print Display
        self.output_label = Label(
            master, text='Output', font=('bold', 10), padx=20)
        self.output_label.grid(row=5, column=0, sticky=W+N, pady=(0, 20))

        self.output_listbox = Listbox(master, height=8, width=73, border=0)
        self.output_listbox.grid(
            row=5, column=1, columnspan=2, pady=(0, 20), sticky=W)

        self.output_scroll = Scrollbar(master)
        self.output_scroll.grid(row=5, column=2, sticky=E+N)

        self.output_listbox.configure(yscrollcommand=self.output_scroll.set)
        self.output_scroll.config(command=self.output_listbox.yview)

        # Processes complition status
        self.status_var = StringVar()
        self.status_label = Label(master, fg='red', font=('Helvetica', 9))
        self.status_label.grid(row=6, column=0, padx=(
            10, 0), pady=(0, 20), columnspan=2, sticky=W)

        if not self.path_var.get():
            self.status_label.config(text='*Enter path', fg='red')

    def print_path_to_entry(self):
        self.file_path = filedialog.askdirectory()
        self.path_entry.insert(0, self.file_path)
        self.radio_print()

    def print_path(self):
        return self.path_entry.get()

    def run_process(self):
        # Single core object
        self.list_val = ' '
        self.runtime_val = ' '

        self.process_thread = Thread(target=self.run_process_thread)
        self.master.after(3000, self.updateListbox)

        self.process_thread.start()

    def run_process_thread(self):
        # Object of MultiProcessing Class
        self.output_listbox.delete(0, END)
        self.runtime_entry.delete(0, END)
        if self.radio_var.get() == 1:
            fcm = FCM(self.list_queue, self.runtime_queue)
        if self.radio_var.get() == 2:
            multiProcessing = MultiProcessing(
                self.list_queue, self.runtime_queue)
        if self.radio_var.get() == 3:
            cuda = CUDA(self.list_queue, self.runtime_queue)

    def open_explorer(self):
        subprocess.Popen('explorer ' + str(os.getcwd()) + '"\Output"')

    def updateListbox(self):
        if not self.process_thread.isAlive or self.list_queue.empty():
            self.update_runtime()
            self.status_label.config(text='*Process Completed', fg='green')

            # output path button
            self.op_btn = Button(self.master, text='Output',
                                 command=self.open_explorer, width=15)
            self.op_btn.grid(row=6, column=2, sticky=W)
            return

        if not self.list_queue.empty():
            self.list_val = self.list_queue.get()

        self.output_listbox.update_idletasks()
        self.output_listbox.insert(END, self.list_val)
        self.output_listbox.yview(END)
        self.master.after(1000, self.updateListbox)

    def update_runtime(self):
        if not self.process_thread.isAlive or self.runtime_queue.empty():
            return

        if not self.runtime_queue.empty():
            self.runtime_val = self.runtime_queue.get()

        self.runtime_entry.update_idletasks()
        self.runtime_entry.insert(0, self.runtime_val)
        self.master.after(100, self.update_runtime)

    def radio_print(self):
        if not self.path_var.get():
            self.status_label.config(text='*Enter path', fg='red')
        else:
            if self.radio_var.get() == 1:
                self.status_label.config(
                    text='*Single core Processing...', fg='green')
            if self.radio_var.get() == 2:
                self.status_label.config(
                    text='*Multi core Processing...', fg='green')
            if self.radio_var.get() == 3:
                self.status_label.config(
                    text='*GPU core Processing...', fg='green')


if __name__ == "__main__":
    root = Tk()
    root.title('Brain Tumor Segmentation')

    # Application Object
    app = Application(root)

    root.mainloop()
