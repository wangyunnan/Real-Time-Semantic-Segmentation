import numpy as np
import cv2
import os
import os.path as osp


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_sg.avi', fourcc, 15, (1280, 720), True)
for i in range(6368):
    vpath = '/media/wangyunnan/HP P500/deep learning/zzz2/front'
    vpath = osp.join(vpath, '{num:d}'.format(num=i) + '.jpg')
    print(vpath)
    orign = cv2.imread(vpath)
    out.write(orign)
    print(i)

out.release()
cv2.destroyAllWindows()