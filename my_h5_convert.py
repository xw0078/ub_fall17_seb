#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------------------------------
Description:    Convert H5 data
Version:        v1
Author:         Xinyue Wang
Date:           xxx

Usage:

Input:
Output:

------------------------------------------------------------------------------------------------'''

import sys
import os
import re
import argparse
import codecs
import h5py
import numpy as np
import datetime


# SETTINGS
target_channels = ['1W-2_12','1W-2_03','1W-2_15']
sample_rate = 256 # 256 samples per second
one_hour_sample = 921600 # 921600 data per hour
lable_period = 


def main():
    '''Main
    '''
    input_dir = arguments_handler() # handle arguments
    process_H5(input_dir) # main process

def arguments_handler():
    '''Handle arguments
    '''
    parser = argparse.ArgumentParser(description = 'Convert H5 data')
    parser.add_argument('-i', '--input-dir', help = 'Input dirctory for a list of h5 data', required = True)
    args = parser.parse_args()
    input_dir = args.input_dir
    return input_dir

def get_timestamp_array():
    '''get timestamp from the dataset
        start time read from file name
    '''
    timestamp_array = np.array(range(one_hour_sample))
    print(timestamp_array.shape)
    return timestamp_array

#timestamp_row = get_timestamp_array() # [0,1,...,921599]

def get_1s_time_array():
    timestamp_array = np.array(range(0,3600))
    print(timestamp_array.shape)
    return timestamp_array

timestamp_row = get_1s_time_array() # [0,1,...,3559]

def process_H5(input_dir):
    '''Process controller
    '''
    # iterate target directory
    for f in os.listdir(input_dir):
         if f.endswith(".h5"):
            file_path = os.path.join(input_dir, f)
            file_date = f.replace(".h5","")
            dataset = readH5File(file_path)
            out_name = f.replace(".h5",".seb")
            out_file = os.path.join(input_dir, out_name)
            outputData(out_file,dataset,file_date)


def readH5File(input_path):
    
    '''Read H5 file
    args:
        input_path (string)
    '''
    # load file content to h5py obj
    my_h5_file = h5py.File(input_path, 'r')
    # extract target channels into matrix
    flag = 0
    for ch in target_channels:
        new_row = extract_channel_data(ch, my_h5_file)
        if flag == 0:# initialize first row
            my_dataset = new_row
            flag = 1
        else:
            my_dataset = np.vstack((my_dataset,new_row))
        print("ch:", my_dataset.shape)
    # map timestamp data
    print("time:", timestamp_row.shape)
    my_dataset = np.vstack((my_dataset,timestamp_row))
    print("final:", my_dataset.shape)
    # add lable to data
    
    # transpose the matrix (column based data)
    my_dataset = np.transpose(my_dataset)
    print(my_dataset.shape)
    return my_dataset
    

def extract_channel_data(channel_name, h5_file):
    '''extract data from input channel name
        return a flatten row
    '''
    dataset = h5_file['/Data/Data_'+channel_name]

    # extract plain flattened data
    extracted_data = np.asarray(dataset).flatten()
    # get average data on seconds (3600sec per hour)
    data_by_sec = []
    abs_1s = 0
    for i in range(0,3600):
        for j in range(0,256):
            position = i*256+j
            abs_1s = abs_1s + abs(extracted_data[position])
        abs_1s = abs_1s/256
        data_by_sec.append(abs_1s)
    # convert to np array
    data_by_sec = np.asarray(data_by_sec, dtype = np.float32)
    print(data_by_sec.shape)
    return data_by_sec
        
def outputData(output_path,dataset,file_date):
    ''' Output data
    '''
    out_file = open(output_path,'w+') # reading/writing, overwrite exist one
    # write header into file
    '''
    for item in target_channels:
        out_file.write(item+' ')
    out_file.write("time_step date_hour\n")
    '''
    # write data into file
    for i in range(0,3600):
        for j in range(0,len(target_channels)+1):
            out_file.write(str(dataset[i][j])+' ')
        out_file.write(file_date+'\n')

# call main function
if __name__ == "__main__":
    main()
