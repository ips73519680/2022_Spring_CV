import numpy as np


def solve_homography(u, v):
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    # u=((ux1,uy1),(ux2,uy2),(ux3,uy3),(ux4,uy4))
    # v=((vx1,vy1),(vx2,vy2),(vx3,vy3),(vx4,vy4))
    ux=u[:,0].reshape((N,1)) # (ux1,ux2,ux3,ux4) !! should be 2D array !!
    uy=u[:,1].reshape((N,1)) # (uy1,uy2,uy3,uy4) 
    vx=v[:,0].reshape((N,1))
    vy=v[:,1].reshape((N,1))

    constraint1_A = np.hstack([np.zeros((N,3)),ux,uy,np.ones((N,1)),-1*np.multiply(vy,ux),-1*np.multiply(vy,uy),-1*vy])
    constraint2_A = np.hstack([ux,uy,np.ones((N,1)),np.zeros((N,3)),-1*np.multiply(vx,ux),-1*np.multiply(uy,vx),-1*vx])
    A= np.vstack ([constraint1_A,constraint2_A])

    # TODO: 2.solve H with A
    U, S, VH = np.linalg.svd(A)

    # Let ℎ be the last column of 𝑉 , VH[-1,-1] = h9
    H=VH[-1,:]/VH[-1,-1]
    H=H.reshape((3,3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xx,yy=np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    # [x1  x2  x3 ...]
    # [y1  y2  y3 ...]
    # [1   1   1  ...]
    x=np.expand_dims(xx.flatten(),0)
    y=np.expand_dims(yy.flatten(),0)
    one=np.ones((1,x.shape[1]))
    M = np.concatenate((x, y, one), axis = 0)

    # from given  (x,y) coordinate pairs destination(M)  to source
    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        M_transformed  = np.dot(H_inv,M) # homogeneous coordinate
        M_transformed_Ordinary = np.round(np.divide(M_transformed[:2], M_transformed[-1,:])).astype(int)  # Ordinary Coordinate
        x_source=M_transformed_Ordinary[0].reshape((ymax-ymin, xmax-xmin))
        y_source=M_transformed_Ordinary[1].reshape((ymax-ymin, xmax-xmin))
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
         # generate 2D boolean array for height and width
        h_mask = (0<y_source)*(y_source<h_src) 
        w_mask = (0<x_source)*(x_source<w_src)
        # if both truth :mask = truth
        mask   = h_mask*w_mask

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        x_source_masked=x_source[mask]
        y_source_masked=y_source[mask]

        # TODO: 6. assign to destination image with proper masking
        dst[yy[mask],xx[mask]] = src[y_source_masked, x_source_masked]
        pass

    # from given  (x,y) coordinate pairs source(M) to  destination
    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        M_transformed  = np.dot(H,M) # homogeneous coordinate
        M_transformed_Ordinary = (np.round(np.divide(M_transformed[:2], M_transformed[-1,:]))).astype(int)  # Ordinary Coordinate
        x_destination=M_transformed_Ordinary[0].reshape((ymax-ymin, xmax-xmin))
        y_destination=M_transformed_Ordinary[1].reshape((ymax-ymin, xmax-xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)     
        # TODO: 5.filter the valid coordinates using previous obtained mask
        # if exceed the boundaries then clip them
        np.clip(x_destination, 0, dst.shape[1]-1)
        np.clip(y_destination, 0, dst.shape[0]-1)

        # TODO: 6. assign to destination image using advanced array indicing
        dst[y_destination, x_destination] = src
        pass

    return dst
