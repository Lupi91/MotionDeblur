# source: https://github.com/opencv/opencv/blob/5199850039ad23f1f0e6cccea5061a9fea5efca6/samples/python/deconvolution.py
'''
Wiener deconvolution.

Sample shows how DFT can be used to perform Weiner deconvolution [1]
of an image with user-defined point spread function (PSF)

Usage:
  deconvolution.py  [--circle]
      [--angle <degrees>]
      [--d <diameter>]
      [--snr <signal/noise ratio in db>]
      [<input image>]

  Use sliders to adjust PSF paramitiers.
  Keys:
    SPACE - switch btw linear/circular PSF
    ESC   - exit

Examples:
  deconvolution.py --angle 135 --d 22  licenseplate_motion.jpg
    (image source: http://www.topazlabs.com/infocus/_images/licenseplate_compare.jpg)

  deconvolution.py --angle 86 --d 31  text_motion.jpg
  deconvolution.py --circle --d 19  text_defocus.jpg
    (image source: compact digital photo camera, no artificial distortion)


[1] http://en.wikipedia.org/wiki/Wiener_deconvolution
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv.copyMakeBorder(img, d, d, d, d, cv.BORDER_WRAP)
    img_blur = cv.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv.warpAffine(kern, A, (sz, sz), flags=cv.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz):
    kern = np.zeros((sz, sz), np.uint8)
    cv.circle(kern, (sz, sz), d, 255, -1, cv.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern


def main():
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['circle', 'angle=', 'd=', 'snr='])
    opts = dict(opts)
    try:
        fn = args[0]
    except Exception as e:
        print("Error loading image!")
        print(e)
        sys.exit(1)

    win = 'deconvolution'

    img = cv.imread(cv.samples.findFile(fn), cv.IMREAD_GRAYSCALE)
    if img is None:
        print('Failed to load file:', fn)
        sys.exit(1)

    img = np.float32(img)/255.0
    cv.imshow('input', img)

    img = blur_edge(img)
    IMG = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)

    defocus = '--circle' in opts

    def update(_):
        ang = np.deg2rad( cv.getTrackbarPos('angle', win) )
        d = cv.getTrackbarPos('dist', win)
        noise = 10**(-0.1*cv.getTrackbarPos('SNR (db)', win))
        sz = cv.getTrackbarPos('kernel', win)

        if defocus:
            psf = defocus_kernel(d, sz)
        else:
            psf = motion_kernel(ang, d, sz)
        cv.imshow('psf', psf)

        psf /= psf.sum()
        psf_pad = np.zeros_like(img)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv.dft(psf_pad, flags=cv.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
        PSF2 = (PSF**2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
        RES = cv.mulSpectrums(IMG, iPSF, 0)
        res = cv.idft(RES, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT )
        res = np.roll(res, -kh//2, 0)
        res = np.roll(res, -kw//2, 1)
        cv.imshow(win, res)

    cv.namedWindow(win)
    cv.namedWindow('psf', 0)
    cv.createTrackbar('angle', win, int(opts.get('--angle', 0)), 180, update)
    cv.createTrackbar('dist', win, int(opts.get('--d', 25)), 200, update)
    cv.createTrackbar('SNR (db)', win, int(opts.get('--snr', 20)), 100, update)
    # adjust the kernel size (was hardcoded as 65)
    cv.createTrackbar('kernel', win, int(opts.get('--k', 60)), 100, update)
    update(None)

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            #defocus = not defocus
            # start with 'linear' as default
            defocus = defocus
            update(None)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()