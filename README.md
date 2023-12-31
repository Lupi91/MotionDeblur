# MotionDeblur
Example usage of wiener deconvolution to restore motion-blurred images.<br>
- Original code: https://github.com/opencv/opencv/blob/5199850039ad23f1f0e6cccea5061a9fea5efca6/samples/python/deconvolution.py
- Changes: Added a trackbar to modify the kernel size

**Requirements**
- cv2
- numpy

**Usage**
```
python main.py <path_to_image>
```

## Restoring motion-blurred license plates
![](/Demo.png?raw=true)


## References
Motion Deblur Filter: https://docs.opencv.org/3.4/d1/dfd/tutorial_motion_deblur_filter.html
