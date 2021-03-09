# coding: utf-8

def same_padding(size_kernel, stride):
    padding = (size_kernel-stride)/2
    if not padding.is_integer():
        padding += 1        
    return int(padding)