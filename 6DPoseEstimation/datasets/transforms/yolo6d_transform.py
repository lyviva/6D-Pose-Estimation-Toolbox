import os
import random
import numpy as np
from PIL import Image, ImageMath
from mmyolo.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

def change_background(img, mask, bg):
        ow, oh = img.size
        bg = bg.resize((ow, oh)).convert('RGB')
        
        imcs = list(img.split())
        bgcs = list(bg.split())
        maskcs = list(mask.split())
        fics = list(Image.new(img.mode, img.size).split())
        
        for c in range(len(imcs)):
            negmask = maskcs[c].point(lambda i: 1 - i / 255)
            posmask = maskcs[c].point(lambda i: i / 255)
            fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
        out = Image.merge(img.mode, tuple(fics))
        
        return out


def data_augmentation(img, shape, jitter, hue, saturation, exposure):

    ow, oh = img.size
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res  = distort_image(im, dhue, dsat, dexp)
    return res

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im


def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale


@TRANSFORMS.register_module()
class CopyPaste6D(BaseTransform):
    """change the background"""
    def __init__(self,
                 shape,
                 jitter,
                 hue,
                 saturation,
                 exposure,
                 num_keypoints,
                 max_num_gt
    ):
        self.shape = shape
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.num_keypoints = num_keypoints
        self.max_num_gt = max_num_gt

    def transform(self, results:dict) -> dict:
        ## data augmentation
        imgpath = results['img_path']
        maskpath = results['mask_path']
        bgpath = results['bg_path']
        
        img = Image.open(imgpath).convert('RGB')
        mask = Image.open(maskpath).convert('RGB')
        bg = Image.open(bgpath).convert('RGB')
        
        img = change_background(img, mask, bg)
        img,flip,dx,dy,sx,sy = data_augmentation(img, self.shape, self.jitter,
                                                 self.hue, self.saturation,
                                                 self.exposure)
        results = self.fill_truth_detection(results, dx, dy, 1./sx, 1./sy,
                                          self.num_keypoints, self.max_num_gt)
        
        return results
    
    def fill_truth_detection(self, results, dx, dy, sx, sy, num_keypoints):
        cc = 0
        for i, instance in enumerate(results['instances']):
            
            bs = np.stack([instance['center_norm'].unsqueeze(), 
                           instance['corners_norm']])
            xs = list()
            ys = list()
            for j in range(num_keypoints):
                xs.append(bs[2*j])
                ys.append(bs[2*j+1])

            # Make sure the centroid of the object/hand is within image
            xs[0] = min(0.999, max(0, xs[0] * sx - dx)) 
            ys[0] = min(0.999, max(0, ys[0] * sy - dy)) 
            
            for j in range(1,num_keypoints):
                xs[j] = xs[j] * sx - dx 
                ys[j] = ys[j] * sy - dy 
            
            for j in range(num_keypoints):
                bs[2*j] = xs[j]
                bs[2*j+1] = ys[j]
            
            results['instances'][i]['center_norm'] = bs[0, :]
            results['instances'][i]['corners_norm'] = bs[1:num_keypoints, :]

        return results