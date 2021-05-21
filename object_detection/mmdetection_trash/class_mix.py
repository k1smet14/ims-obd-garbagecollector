from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import json
import sys
import copy
import glob

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

color = [(0,0,0), (255,255,0), (0,255,0), (0,0,255),
     (0,255,255), (255,0,255), (128,0,255),(128,128,128),
    (255,128,0),(0,255,128),(255,0,0)]

def get_cate_idx(coco,idx):
    cate_idxs = []
    for i in range(len(coco.getImgIds())):
        ann_ids = coco.getAnnIds(i)
        ann_info = coco.loadAnns(ann_ids)

        for info in ann_info:
            if info['category_id'] == idx:
                cate_idxs.append(i)
                break

    return cate_idxs


def draw_box(img,ann_info,idx2name):
    view_img = img.copy()
    bbox_thick = int(0.6 * (512 + 512) / 600)
    fontScale = 0.5
    
    for info in ann_info:
        cls_name = idx2name[info['category_id']]
        box_color = color[info['category_id']]
        x1, y1, w, h = info['bbox']
        view_img = cv2.rectangle(view_img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),box_color,2)
        t_size = cv2.getTextSize(cls_name, 0, fontScale, thickness=bbox_thick // 2)[0]
        t = (int(x1 + t_size[0]), int(y1 - t_size[1] - 3))
        view_img = cv2.rectangle(view_img, (int(x1),int(y1)), (t[0], t[1]), box_color, -1)
        view_img = cv2.putText(view_img, cls_name, (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return view_img


def load_back(coco,idx,data_root,idx2name,is_remove=False):
    img_info = coco.loadImgs(idx)[0]
    img  = cv2.imread(os.path.join(data_root,img_info['file_name']))
    ann = []
    view_img = img.copy()
    if not is_remove:
        ann_ids = coco.getAnnIds(idx)
        ann_info = coco.loadAnns(ann_ids)
        
        bbox_thick = int(0.6 * (512 + 512) / 600)
        fontScale = 0.5
        
        for info in ann_info:
            cls_name = idx2name[info['category_id']]
            box_color = color[info['category_id']]
            x1, y1, w, h = info['bbox']
            view_img = cv2.rectangle(view_img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),box_color,2)
            t_size = cv2.getTextSize(cls_name, 0, fontScale, thickness=bbox_thick // 2)[0]
            t = (int(x1 + t_size[0]), int(y1 - t_size[1] - 3))
            view_img = cv2.rectangle(view_img, (int(x1),int(y1)), (t[0], t[1]), box_color, -1)
            view_img = cv2.putText(view_img, cls_name, (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        ann = ann_info
    return img, view_img, ann


def load_front(coco,idx,data_root,cate_idx):
    img_info = coco.loadImgs(idx)[0]
    img  = cv2.imread(os.path.join(data_root,img_info['file_name']))
    ann = []
    ann_ids = coco.getAnnIds(idx)
    ann_info = coco.loadAnns(ann_ids)
    masks = None
    for info in ann_info:
        if info['category_id'] == cate_idx:
            mask = coco.annToMask(info)
            mask = np.maximum(mask*(cate_idx+1), mask)
            if masks is None:
                masks = mask
            else:
                masks += mask

            ann.append(info)
    return img, ann, masks


def affine_front(img,masks,ann,t_w,t_h,rot,s,is_visualize=False):
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=rot, center=(256, 256), scale=s)
    
    T = np.eye(3)
    T[0, 2] = t_w
    T[1, 2] = t_h

    M = T@R

    img = cv2.warpAffine(img, M[:2], dsize=(512,512), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    masks = cv2.warpAffine(masks, M[:2], dsize=(512,512), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    new_ann =copy.deepcopy(ann) #.copy()
    
    for info in new_ann:

        # calculate new box
        x1, y1, w, h = info['bbox']
        nx1,ny1, _ = M @ [x1,y1,1] 
        nx2,ny2, _= M @ [x1+w,y1+h,1]
        xs,ys,xl,yl =min(nx1,nx2),min(ny1,ny2),max(nx1,nx2),max(ny1,ny2)

        n_box = np.array([xs,ys,xl,yl]).clip(0,511)

        if (n_box[2]-n_box[0]) * (n_box[3]-n_box[1]) < 100:
            continue
        
        # assign new box
        info['bbox'] = [n_box[0],n_box[1],n_box[2]-n_box[0],n_box[3]-n_box[1]]


        # assign new segmentation
        n_seg = []
        for seg in info['segmentation']:
            seg = np.array(seg).reshape(-1,2)
            seg = np.concatenate([seg,np.ones((seg.shape[0],1))],-1)
            seg = np.matmul(M,seg.transpose())
            seg =  list(np.int32(seg)[:2].transpose().reshape(-1).clip(0,511))
            n_seg.append(seg)
        info['segmentation'] = n_seg

    # for debug
    if is_visualize:
        view_img = img.copy()
        box_color = (255,0,0)
        for info in new_ann:
            x1, y1, w, h = info['bbox']
            view_img = cv2.rectangle(view_img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),box_color,2)
            
        plt.imshow(view_img)
        plt.show()

    return img, new_ann, masks



def synthesis_img(back_img,back_view_img,front_img,masks):
    tile_mask = np.tile(np.expand_dims(masks,-1),[1,1,3])
    syn_img = np.where(tile_mask>0,front_img,back_img)
    syn_view_img = np.where(tile_mask>0,front_img,back_view_img)
    return syn_img,syn_view_img


def make_mix_data(args):
    data_root = args.data_root
    json_name = args.json_name
    synthesis_folder = args.synthesis_folder
    synthesis_path = os.path.join(data_root,synthesis_folder)

    with open(os.path.join(data_root,json_name), 'r') as f:
        ori_json = json.load(f)

    # dict_keys(['info', 'licenses', 'images', 'categories', 'annotations'])
    if args.is_new:
        new_json = dict()
        for k in ['info','licenses','categories']:
            new_json[k] = ori_json[k]
        new_json['images'] = []
        new_json['annotations'] = []
    else:
        new_json = copy.deepcopy(ori_json)

    if not os.path.exists(os.path.join(data_root,synthesis_folder)):
        os.makedirs(os.path.join(data_root,synthesis_folder))

    print('Load coco.....')
    coco = COCO(os.path.join(data_root,json_name))
    classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    name2idx = {c: i for i,c in enumerate(classes)}
    idx2name = {i : c for i,c in enumerate(classes)}

    exit_flag=False
    is_remove = args.is_remove

    now_ann_id = 0 if args.is_new else len(coco.getAnnIds())
    now_image_id = 0 if args.is_new else len(coco.getImgIds())
    back_id,front_id = 0,0

    t_scale = 10
    t_h,t_w,rot,s = 0, 0, 0, 1.
    is_add = False

    try:
        while not exit_flag:
            user_input = input(f'Input { [c for c in classes]} or 0~10  exit=q : ')
            
            if user_input in name2idx.keys():
                cate_idx = name2idx[user_input]
            elif user_input =='q':
                break
            else:
                try:
                    cate_idx = int(user_input)
                except:
                    print("input is wrong!")
                    continue
            print(f"Your selection is {idx2name[cate_idx]}")
            exit_flag=True
            
            cate_idxs = get_cate_idx(coco,cate_idx)
            other_idxs = list(set(coco.getAnnIds()) - set(cate_idxs))
            
            back_id = min(back_id,len(other_idxs)-1)
            front_id = min(front_id,len(cate_idxs)-1)
            if not is_add:
                back_img, back_view_img, back_ann = load_back(coco,other_idxs[back_id],data_root,idx2name,is_remove)
            else:
                is_add = False

            front_img, front_ann, masks = load_front(coco,cate_idxs[front_id],data_root,cate_idx)
            now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s,False)

            syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
            
            
            while True:
                print("---------------------key infor mation----------------------")
                print("[8456] : translate, [79] : adjust translate scale, [+-] : object scale\n[r] : rotation, [df] : select back img, [cv] : select object img\n[s] : save, [q] : exit, [a] : now image to back img, [o] : set default")
                print("-"*30)
                cv2.imshow('synthesis_img',syn_view_img)
                key = cv2.waitKey()

                if key==ord('8'):
                    t_h -= t_scale
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key==ord('4'):
                    t_w -= t_scale
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key==ord('5'):
                    t_h += t_scale
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key==ord('6'):
                    t_w += t_scale
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key==ord('r'):
                    rot += 90
                    if rot==360:
                        rot = 0
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key==ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif key==ord('s'):
                    num = len(glob.glob(synthesis_path+'/*.jpg'))
                    
                    cv2.imwrite(os.path.join(synthesis_path,f'{num:04d}.jpg'),syn_img)
                    new_json['images'].append({'license': 0,
                        'url': None,
                        'file_name': f'{synthesis_folder}/{num:04d}.jpg',
                        'height': 512,
                        'width': 512,
                        'date_captured': None,
                        'id': now_image_id})
                    
                    new_ann = copy.deepcopy(back_ann) + copy.deepcopy(now_front_ann)
                    for i in range(len(new_ann)):
                        new_ann[i]['id'] = now_ann_id
                        new_ann[i]['image_id'] = now_image_id
                        now_ann_id += 1
                    now_image_id +=1
                    new_json['annotations'].extend(new_ann)
                    print(f"saved image id ={now_image_id-1}, ann id = {now_ann_id-1}")
                    print(f'{synthesis_folder}/{num:04d}.jpg saved!')
                elif key==ord('d'):
                    if back_id == 0:
                        continue
                    back_id -= 1
                    back_img, back_view_img, back_ann = load_back(coco,other_idxs[back_id],data_root,idx2name,is_remove)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key==ord('f'):
                    if back_id == len(other_idxs) -1:
                        continue
                    back_id += 1
                    back_img, back_view_img, back_ann = load_back(coco,other_idxs[back_id],data_root,idx2name,is_remove)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key==ord('c'):
                    if front_id == 0:
                        continue
                    front_id -= 1
                    front_img, front_ann, masks = load_front(coco,cate_idxs[front_id],data_root,cate_idx)
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key==ord('v'):
                    if front_id == len(cate_idxs)-1:
                        continue
                    front_id += 1
                    front_img, front_ann, masks = load_front(coco,cate_idxs[front_id],data_root,cate_idx)
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key == ord('+'):
                    s+=0.1
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key == ord('-'):
                    s-=0.1
                    if s<0:
                        s=0.
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)
                elif key == ord('7'):
                    t_scale -= 1
                    t_scale = max(0,t_scale)
                    print(f'translate_scale = {t_scale}')
                elif key == ord('9'):
                    t_scale += 1
                    print(f'translate_scale = {t_scale}')
                elif key == ord('a'):
                    back_img = syn_img.copy()
                    back_ann = back_ann + now_front_ann
                    back_view_img = draw_box(back_img,back_ann,idx2name)
                    exit_flag = False
                    is_add = True
                    break
                elif key == ord('o'):
                    t_scale = 10
                    t_h,t_w,rot,s = 0, 0, 0, 1.
                    now_front, now_front_ann, now_masks = affine_front(front_img,masks,front_ann,t_w,t_h,rot,s)
                    syn_img, syn_view_img = synthesis_img(back_img, back_view_img, now_front, now_masks)

        print('Make json.....')
        with open(os.path.join(data_root,args.out_json_name), 'w', encoding='utf-8') as f:
            json.dump(new_json, f, indent="\t",cls=NpEncoder)
        print(f'{os.path.join(data_root,args.out_json_name)} saved!')
    except:
        print('Make json.....')
        with open(os.path.join(data_root,args.out_json_name), 'w', encoding='utf-8') as f:
            json.dump(new_json, f, indent="\t",cls=NpEncoder)
        print(f'{os.path.join(data_root,args.out_json_name)} saved!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Make mix dataset')
    parser.add_argument('-d','--data_root',              type=str,   help='Root path of json and images', default='input/data')
    parser.add_argument('-j', '--json_name',              type=str,   help='Name of json file', default='train_all.json')
    parser.add_argument('-s','--synthesis_folder',              type=str,   help='Folder of result images', default='synthesis_song')
    parser.add_argument('-o','--out_json_name',              type=str,   help='Name of json file', default='train_synthesis_song_all.json')
    parser.add_argument('-r','--is_remove',              action='store_true',   help='Remove first background ann')
    parser.add_argument('-n','--is_new',              action='store_true',   help='Remove first background ann')
    args = parser.parse_args()

    make_mix_data(args)