
import cv2
import numpy as np
import cv2.ximgproc as xip


def computeLBP(img,ch,x,y):
    img=np.array(img)
    LBP_arr=[]
    LBP=[]

    center=img[y][x]

    #shape =(8,ch)
    LBP_arr.append(img[y+1][x-1])     # top_right
    LBP_arr.append(img[y+1][x])       # right
    LBP_arr.append(img[y+1][x+1])     # bottom_right
    LBP_arr.append(img[y][x+1])       # bottom
    LBP_arr.append(img[y-1][x+1])     # bottom_left
    LBP_arr.append(img[y-1][x])       # left
    LBP_arr.append(img[y-1][x-1])     # top_left
    LBP_arr.append(img[y][x-1])       # top
    
    LBP_arr=np.array(LBP_arr)

    LBP_slice=[]
    for i in range(ch):
        LBP_slice=LBP_arr[...,i]
        LBP_slice[LBP_slice>center[i]]=1
        LBP_slice[LBP_slice!=1]=0
        LBP.append(LBP_slice)

    LBP=np.array(LBP)

    return LBP


def computeDisp(Il, Ir, max_disp):

    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
     # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    Il_pad = cv2.copyMakeBorder(Il, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0,0,0))
    Ir_pad = cv2.copyMakeBorder(Ir, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0,0,0))
    LBP_IL=np.zeros((h,w,ch,8))
    LBP_IR=np.zeros((h,w,ch,8))
    
    for x in (range((w))):
        for y in range(h):
            LBP_IL[y][x]=computeLBP(img=Il_pad,ch=ch,x=x+1,y=y+1)
            LBP_IR[y][x]=computeLBP(img=Ir_pad,ch=ch,x=x+1,y=y+1)

    cost_map_L2R=np.ones(((h,w,max_disp+1)), dtype=np.float32)*24
    cost_map_R2L=np.ones(((h,w,max_disp+1)), dtype=np.float32) *24

    for disp in (range(max_disp+1)):
            if(disp==0):
                cost_map_L2R[...,disp] = np.count_nonzero(LBP_IL!=LBP_IR,axis=(2,3))
                cost_map_R2L[...,disp] = np.count_nonzero(LBP_IR!=LBP_IL,axis=(2,3))
            else:
                cost_map_L2R[:,disp:,disp] = np.count_nonzero(LBP_IL[:,disp:,:,:]!=LBP_IR[:,:-disp,:,:],axis=(2,3))
                cost_map_R2L[:,:-disp,disp] = np.count_nonzero(LBP_IR[:,:-disp,:,:]!=LBP_IL[:,disp:,:,:],axis=(2,3))
            cost_map_L2R[...,disp]=xip.jointBilateralFilter(Il,cost_map_L2R[...,disp],10,10,10)
            cost_map_R2L[...,disp]=xip.jointBilateralFilter(Ir,cost_map_R2L[...,disp],10,10,10)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    Winner_cost_L2R=np.argmin(cost_map_L2R,axis=2)
    Winner_cost_R2L=np.argmin(cost_map_R2L,axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering 
    for x in range(w):
        for y in range(h):
            if (Winner_cost_L2R[y,x] != Winner_cost_R2L[y,x-Winner_cost_L2R[y,x]]):
                # print(x,',',y)
                Winner_cost_L2R[y,x]=-1

    for x in range(w):
        for y in range(h):           
            FR=max_disp
            FL=max_disp
            if (Winner_cost_L2R[y,x]==-1):
                #left
                for left in range(max_disp+1):
                    if( x-left < 0 ):break
                    elif( Winner_cost_L2R[y,x-left]==-1):continue
                    else:
                        FL=Winner_cost_L2R[y,x-left]
                        break
                #right
                for right in range(max_disp+1):
                    if( x+right == w ):break
                    elif( Winner_cost_L2R[y,x+right]==-1):continue
                    else:
                        FR=Winner_cost_L2R[y,x+right]
                        break
                Winner_cost_L2R[y,x] = min(FL, FR)    


    # print('total time: ({}min {}sec )'.format((end_time-start_time)//60,(end_time-start_time)%60))

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), Winner_cost_L2R.astype(np.uint8), 18, 1)

    return labels.astype(np.uint8)
    

#     tsu 

# 6.09% >> 5,10,10
# 3.49% >> 10,10,10
# 3.83% >> 15,10,10
# 4.84  >> 30 10 10
# 5.46  >> 30 20 20


# teddy

# 12.7 >>10 5 5 
# 9.99 >>30 10 10
# 11.20 >>10 10 10
# 11.15 >> 15 10 10


