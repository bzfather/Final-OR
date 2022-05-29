
import mmcv
import os
import os.path as osp
import numpy as np
class Estimator_2d (  ):

    def __init__(self, DEBUGGING=False):
        self.bbox_detector =DEBUGGING

        
    def get_2d(self,cam_id,frame_id):
        ##load 2d results from files

        cam=cam_id+1
        pathPos="datasets/realor/poses/cam"+str(cam)
        model_dir = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ) ) )
        root_dir = os.path.abspath ( os.path.join ( model_dir, '..' ) )        
        path=osp.join(root_dir,pathPos)
        img = osp.join(path, f'{frame_id:06d}.json')
        persons=mmcv.load(img)
        persons=self.deduplication(persons)


    
    
        
        return persons

    def deduplication(self,persons):
        # remove same pose
        
        N = len(persons)
        newperson=[]
        check_dic={}
        ioumat=np.zeros((N,N),dtype=float)
        for i in range(N):
            if (i in check_dic)==True:
                continue

            for j in range(i+1, N):
                boxA = persons[i]['bbox']
                boxB = persons[j]['bbox']
                iou=self.IoU(self.bboxformtran(boxA) ,self.bboxformtran(boxB) )
                ioumat[i][j]=iou
                ##if two detections are overlapping
                if iou> 0.2:

                    areaA=boxA[2]*boxA[3]
                    areaB = boxB[2] * boxB[3]
                    pose1 = np.array(persons[i]['keypoints'], dtype=float).transpose(1, 0)[:-1]
                    pose2 = np.array(persons[j]['keypoints'], dtype=float).transpose(1, 0)[:-1]
                    pose1 = pose1.transpose(1, 0)
                    pose2 = pose2.transpose(1, 0)
                    simi = 0
                    threshold=2
                    #compute similarity, count how many points are the same

                    for joint in range(9):
                        dist = np.linalg.norm(pose1[joint] - pose2[joint])
                        #print(dist)
                        if dist<threshold:
                            simi+=1
                    if simi>1:
                        if areaA>areaB:
                            check_dic[j]=-1
                        else:
                            check_dic[i]=-1

            if (i in check_dic)==False:
                newperson.append(persons[i])
        return newperson
    def bboxformtran(self,bbox):
    #change the format of bounding boxes
        x,y,w,h=bbox[:-1]
        x1=x
        y1=y
        x2=x+w
        y2=y+h
        return [x1,y1,x2,y2]

    def IoU(self,boxA,boxB):
    ## compute IOU

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        #print(interArea)
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        #print(iou)
        return iou

x=Estimator_2d()
x.get_2d(3,5)


