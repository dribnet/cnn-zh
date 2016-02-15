#!/usr/bin/env python

import os
import glob
import numpy, scipy.misc
from PIL import Image
import argparse

class SingleGntImage(object):
    def __init__(self, f):
        self.f = f

    def read_gb_label(self):
        label_gb = self.f.read(2)

        # check garbage label
        if label_gb.encode('hex') is 'ff':
            return True, None
        else:
            label_uft8 = label_gb.decode('gb18030').encode('utf-8')
            return False, label_uft8

    def read_special_hex(self, length):
        num_hex_str = ""
        
        # switch the order of bits
        for i in range(length):
            hex_2b = self.f.read(1)
            num_hex_str = hex_2b + num_hex_str

        return int(num_hex_str.encode('hex'), 16)

    def convert_to_image(self, image_matrix, size, border):
        coresize = size - 2*border

        I = numpy.zeros((size, size), dtype=numpy.uint8)

        src_image = 255-numpy.array(image_matrix, dtype=numpy.uint8)
        src_shape = src_image.shape

        if(src_shape[0] > src_shape[1]):
            longedge = src_shape[0]
        else:
            longedge = src_shape[1]

        scale_fraction = coresize / float(longedge)

        if(scale_fraction < 1.0):
            src_image = scipy.misc.imresize(src_image, scale_fraction)

        sca_shape = src_image.shape
        offx = int((size - sca_shape[0])/2)
        offy = int((size - sca_shape[1])/2)
        I[offx:offx+sca_shape[0], offy:offy+sca_shape[1]] = src_image

        return I

    def read_single_image(self):
        margin = 2
        size = 64

        # try to read next single image
        try:
            self.next_length = self.read_special_hex(4)
        except ValueError:
            # print "Notice: end of file"
            return None, None, None, None, True

        # read the chinese utf-8 label
        self.is_garbage, self.label = self.read_gb_label()

        # read image width and height and do assert
        self.width = self.read_special_hex(2)
        self.height = self.read_special_hex(2)
        assert self.next_length == self.width * self.height + 10

        # read image matrix
        image_matrix_list = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(self.read_special_hex(1))

            image_matrix_list.append(row)

        # convert to numpy ndarray with size of 64x64 and add margin of 2
        self.image_matrix_numpy = self.convert_to_image(image_matrix_list, size, margin)
        return self.label, self.image_matrix_numpy, \
            self.width, self.height, False

class ReadGntFile(object):
    def __init__(self):
        pass

    def find_file(self, indir):
        file_extend = ".gnt"
        self.file_list = []

        # get all gnt files in the dir
        for file_name in glob.glob(os.path.join(indir, '*.gnt')):
            self.file_list.append(file_name)

        return self.file_list

    def show_image(self, outdir):
        count_file = 0
        count_single = 0
        width_list = []
        height_list = []

        #open all gnt files
        for file_name in self.file_list:
            count_file = count_file + 1
            end_of_image = False
            basename = os.path.basename(file_name)[:-4]
            with open(file_name, 'rb') as f:
                count_within_file = 0
                while not end_of_image:
                    this_single_image = SingleGntImage(f)

                    # get the pixel matrix of a single image
                    label, pixel_matrix, width, height, end_of_image = \
                        this_single_image.read_single_image()
                    
                    if not end_of_image:
                        width_list.append(width)
                        height_list.append(height)
                        outshape = numpy.shape(pixel_matrix)
                        print("save: {0:07d} {1:8s} {2:05d} {3} {4:03d} {5:03d} {6:03d} {7:03d}".format(count_single,
                            basename,count_within_file, label,width, height, outshape[0], outshape[1]))

                        self.save_image(outdir, basename, pixel_matrix, label, count_single, count_within_file)
                        count_within_file = count_within_file + 1
                        count_single = count_single + 1

            # print ("End of file #%i") % (count_file)

    def save_image(self, outdir, basename, matrix, label, count, count_within_file):
        outpath = os.path.join(outdir, basename)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        im = Image.fromarray(matrix)
        name = ("{0}/{1:05d}.png".format(outpath, count_within_file))
        im.save(name)
        # name = ("tmp/test-%s (%i).tiff") % (label, count)
        # im.save(name)

def display_char_image(indir, outdir):
    gnt_file = ReadGntFile()
    file_list = gnt_file.find_file(indir)
    gnt_file.show_image(outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert dreaded gnt files to a buch of pngs")
    parser.add_argument('--indir', dest='indir', type=str, \
        help="Directory root for input", default="input")
    parser.add_argument('--outdir', dest='outdir', type=str, \
        help="Directory root for output", default="output")
    args = parser.parse_args()
    display_char_image(args.indir, args.outdir)
