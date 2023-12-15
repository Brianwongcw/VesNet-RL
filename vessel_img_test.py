from Env import Env_multi_sim_img
import numpy as np
from Vessel_3d import Vessel_3d_sim, create_vessel
from PIL import Image
import math
import cv2
import imutils
from datetime import datetime



def create_configs_rand(num):
    configs=[]
    r_min=30
    r_max=75
    for i in range(num):
        offset=np.random.rand()*np.pi/2
        size_3d=[750,700,450]
        r=np.random.randint(r_min+(r_max-r_min)*i/num,r_min+(r_max-r_min)*(i+1)/num)
        c_x=350
        c_y=np.random.randint(50+r,225)
        c=[c_x,c_y]
        config=(c,r,size_3d,offset)
        configs.append(config)
    return configs

    
def get_slicer(self,center_point,theta):
        # image_3d 3d array
        # center_point the center point of the us probe (np array)
        # theta the angle from x axis to the line, which represents the probe [0,pi]
        # probe_width width of the us probe in pixel, 313 by default (37.5mm)
        poi_tmp=[]
        poi=[]
        rot=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        for i in np.arange(-313//2,313//2):
            p=np.dot(rot,np.transpose(np.array([i,0])))
            p=p+np.array(center_point)
            poi_tmp.append([int(p[0]),int(p[1])])
        for i in poi_tmp:
            if i not in poi:
                poi.append(i)
        image,success=merge_image(poi)
        return image

# import pdb; pdb.set_trace()

configs=create_configs_rand(1)
print(configs)

vessels=[]
vessels.append(Vessel_3d_sim(configs[0],probe_width=313))
x, y = np.random.randint(350, 400), np.random.randint(350, 400)
for i in range(45):
    img,_,_ = vessels[0].get_slicer((x, y), math.pi/2 + math.pi/90*i)
    # img,_,_ = vessels[i].get_slicer((350, 350), 30+10*i)
    # print(configs[i][0])
    n = np.sum(img==1)
    # img[img==1]=255
    img = cv2.resize(img, (640, 640))
    cv2.imshow("vessel", img)
    t = datetime.now()
    if n > 100:
        cv2.imwrite(f'/home/bmt-brian/Desktop/img3/vessel{i}_{t}.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows



'''img = create_vessel(config[0],config[1],config[2])
new_img=Image.fromarray(img)
path = f"/home/bmt-brian/Desktop/img2/{config}.png"
if new_img.mode == "F":
    new_img = new_img.convert('L')
new_img.save(path)'''
