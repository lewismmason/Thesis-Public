# This contains various functions required for constructing cubes etc. These
# are used for both training as well as running the network over volumes
import math
import numpy as np
import tensorflow as tf

# We use indices rather than saving all of the cubes for memory management, since we have the indices we can grab cube data at any time etc


def create_cube_indices_all(data3D, cubes_len_zyx):
    # This generates the indices for cubes to be processed by "create_cubes_from_indices"
    data_z, data_y, data_x = data3D.shape[0], data3D.shape[1], data3D.shape[2]

    num_z = math.ceil(data_z / cubes_len_zyx[0])
    num_y = math.ceil(data_y / cubes_len_zyx[1])
    num_x = math.ceil(data_x / cubes_len_zyx[2])

    return create_cube_indices(data3D, cubes_len_zyx, [num_z, num_y, num_x], [0,0,0])


def create_cube_indices_with_overlap(data3D, cubes_len_zyx, num_cubes_zyx, offset_zyx, overlap_zyx):
    # This creates the indices where cubes overlap in all directions. overlap_zyx is the number of pixels that one side will allow to overlap. only the non-overlapping cubes
    # are to be used for reconstruction later, because it removes edge effects (generally the overlap_zyx should be about half of the receptive feild length, otherwise its tough to make good predictions for those edge pixels)
    # For example, if cubes_len_zyx = [10,10,10] and overlap_zyx = [2,2,2], the non-overlapping parts of the cubes will actually be cybes_len_zyx_real = [6,6,6] (ie thats what we consider to be "good predictions"), for reconstruction
    # you must reconstruct removing remove_zyx = overlap_zyx from the function "add_cubes_to_data_from_indices". This means none of the garbage edge data is used, which we want

    # OUTPUTS a list of arrays with ZYX values. IE [[Z1, Y1, X1], [Z2, Y2, X2], ... etc]
    len_z, len_y, len_x     = cubes_len_zyx[0], cubes_len_zyx[1], cubes_len_zyx[2]
    off_z, off_y, off_x     = offset_zyx[0], offset_zyx[1], offset_zyx[2] # Note z axis throws error?? idk why
    overlap_z, overlap_y, overlap_x = overlap_zyx[0], overlap_zyx[1], overlap_zyx[2]
    num_z, num_y, num_x     = num_cubes_zyx[0], num_cubes_zyx[1], num_cubes_zyx[2]
    data_z, data_y, data_x  = data3D.shape[0], data3D.shape[1], data3D.shape[2]

    flag_z, flag_y, flag_x = False, False, False


    # Clip cubes in zyx directions if needed, slow right now
    print(num_z, num_y, num_y)
    iz = 0   
    while iz < num_z:
        if (off_z + (iz)*len_z - 2*(iz)*overlap_z) >= data_z-len_z and num_z > iz: 
            num_z = iz+1 # add 1 to include the zero
            flag_z = True
            break
        iz += 1
        
    iy = 0
    while iy < num_y:
        if (off_y + (iy)*len_y - 2*(iy)*overlap_y) >= data_y-len_y and num_y > iy: 
            num_y = iy+1 # add 1 to include the zero
            flag_y = True
            break
        iy += 1

    ix = 0
    while ix < num_x:
        if (off_x + (ix)*len_x - 2*(ix)*overlap_x) >= data_x-len_x and num_x > ix: 
            num_x = ix+1 # add 1 to include the zero
            flag_x = True
            break
        ix += 1

    print(num_z, num_y, num_x)
    num_indices = num_z * num_y * num_x

    indices = np.zeros(shape=(num_indices, 3))
    index_i = 0

    for iz in range(0,num_z):
        for iy in range(0,num_y):
            for ix in range(0,num_x):
                ind_z, ind_y, ind_x = 0,0,0 # just initializing for now

                if iz == num_z-1 and flag_z:
                    ind_z = data_z - len_z # Edge cases, fail fast
                else:
                    ind_z = off_z + iz*len_z - 2*iz*overlap_z

                if iy == num_y-1 and flag_y:
                    ind_y = data_y - len_y # Edge cases, fail fast
                else:
                    ind_y = off_y + iy*len_y - 2*iy*overlap_y

                if ix == num_x-1 and flag_x:
                    ind_x = data_x - len_x # Edge cases, fail fast
                else:
                    ind_x = off_x + ix*len_x - 2*ix*overlap_x

                indices[index_i] = [ind_z, ind_y, ind_x]
                index_i = index_i + 1

    return indices


def create_cube_indices(data3D, cubes_len_zyx, num_cubes_zyx, offset_zyx):
    return create_cube_indices_with_overlap(data3D, cubes_len_zyx, num_cubes_zyx, offset_zyx, overlap_zyx=[0,0,0])


def create_cubes_from_indices(data3D, cubes_len_zyx, indices):
    len_z, len_y, len_x = int(cubes_len_zyx[0]), int(cubes_len_zyx[1]), int(cubes_len_zyx[2])
    
    # Greyscale vs RGB
    if data3D.shape[-1] == 3:
        RGB = 3
        cubes = np.zeros(shape=(len(indices), len_z, len_y, len_x, RGB))
    else:
        cubes = np.zeros(shape=(len(indices), len_z, len_y, len_x))

    for i in range(0,len(indices)):
        iz, iy, ix = int(indices[i][0]), int(indices[i][1]), int(indices[i][2])

        cubes[i] = data3D[iz:(iz + len_z), iy:(iy + len_y), ix:(ix + len_x)]

    return cubes


# TODO verify that this works
def create_cube_labels_from_indices(data3D_binary, cubes_len_zyx, indices, padding_index):
    # NOTE: this is troublesome if you want to pad in index 0, idk why you would ever want that. Also can only pad a single time, TODO add array of padding indices, easy.
    # class index is actually the indices to fill with zeros. If padding_index = False, don't pad
    cube_labels = create_cubes_from_indices(data3D_binary, cubes_len_zyx, indices)
    cube_labels = np.clip(cube_labels, 0, 1)                  # Lazy way of binarizing, works because we know the data only has 2 values, 0 and something else
    cube_labels = tf.one_hot(cube_labels, 2)                  # Convert to one-hot for cat crossentropy, "2" means 2 classes. correct as just air and phase, We add other classes below

    # Add zero padding for other classes if required
    if padding_index != False:
        cube_labels = np.insert(cube_labels, padding_index,0, -1)   # Convert to one hot for 3 phases, where there exists none of the newly added phase

    return cube_labels


def add_cubes_to_data_from_indices(new_data3D, cubes, cubes_len_zyx, indices, remove_zyx=[0,0,0]):
    # Given an original tiff, new_tiff is a numpy array of the exact same dimensions. This function populates specific portions of that array
    # based on indices which directly correspond to the cubes in "cubes". Note, cubes and indices must be in order with one another
    # This is used to populate the tiff in batches, because memory will get filled very fast if using total sets of data, cubes, etc etc
    # remove_from_outer removes N of the pixels from the classified image for replacing. This is to counter padding edge effects

    len_z, len_y, len_x = int(cubes_len_zyx[0]), int(cubes_len_zyx[1]), int(cubes_len_zyx[2])
    r_z, r_y, r_x       = int(remove_zyx[0]), int(remove_zyx[1]), int(remove_zyx[2])

    if len(indices) != len(cubes):
        print('Length of indices ' + str(len(indices)) + ' does not match length of cubes ' + str(len(cubes)))

    three_dimensional = len_z != 1 # this determines if squares or cubes, NOT RGB RELATED

    for i in range(0, len(indices)):
        ind_z, ind_y, ind_x = int(indices[i][0]), int(indices[i][1]), int(indices[i][2])
        
        # padding removal included

        if three_dimensional == True:
            new_data3D[ (ind_z + r_z):(ind_z + len_z - r_z), \
                        (ind_y + r_y):(ind_y + len_y - r_y), \
                        (ind_x + r_x):(ind_x + len_x - r_x),...] = \
                        cubes[i, r_z:(len_z-r_z), r_y:(len_y - r_y), r_x:(len_x - r_x),...] # for 3D   

        else:
            new_data3D[ (ind_z + r_z):(ind_z + len_z - r_z), \
                        (ind_y + r_y):(ind_y + len_y - r_y), \
                        (ind_x + r_x):(ind_x + len_x - r_x),...] = \
                        cubes[i, r_y:(len_y - r_y), r_x:(len_x - r_x),...] # for 2D
            
        
    return new_data3D


# NOTE: Square functionality is identical to cubes with z_length = 1, just calls cube functions


# TODO these
def create_square_indices_all(data, squares_len_yx):
    # This generates the indices for cubes to be processed by "create_cubes_from_indices"
    data_z, data_y, data_x = data.shape[0], data.shape[1], data.shape[2]

    num_y = math.ceil(data_y / squares_len_yx[0])
    num_x = math.ceil(data_x / squares_len_yx[1])

    return create_square_indices(data, squares_len_yx, [num_y, num_x], [0,0])


# TODO include z number of squares and offset in z
def create_square_indices(data, squares_len_yx, num_squares_zyx, offset_zyx):
    # NOTE: this is just the same as cubes but with z stuff = 1.
    return create_square_indices_with_overlap(data, squares_len_yx, num_squares_zyx, offset_zyx,overlap_yx=[0,0])

def create_square_indices_with_overlap(data, squares_len_yx, num_squares_zyx, offset_zyx, overlap_yx):
    
    cubes_len_zyx = squares_len_yx.copy()
    cubes_len_zyx.insert(0, 1) # z size is 1

    overlap_zyx = overlap_yx.copy()
    overlap_zyx.insert(0,0) # 0 overlap in z

    return create_cube_indices_with_overlap(data, cubes_len_zyx, num_squares_zyx, offset_zyx, overlap_zyx)


def create_squares_from_indices(data, squares_len_yx, indices):
    cubes_len_zyx = squares_len_yx.copy()
    cubes_len_zyx.insert(0,1)

    squares = create_cubes_from_indices(data, cubes_len_zyx, indices)
    return squares[:,0,:,:] # remove "z" dimension which is just 1 for squares


def create_square_labels_from_indices(data_binary, squares_len_yx, indices, padding_index):
    cubes_len_zyx = squares_len_yx.copy()
    cubes_len_zyx.insert(0,1)

    labels = create_cube_labels_from_indices(data_binary, cubes_len_zyx, indices, padding_index)

    return labels[:,0,:,:]


# TODO add t\e removal functionality to this
def add_squares_to_data_from_indices(new_data, squares, squares_len_yx, indices, remove_yx=[0,0]):
    # TODO check that this works
    cubes_len_zyx = squares_len_yx.copy()
    cubes_len_zyx.insert(0,1)

    remove_zyx = remove_yx.copy()
    remove_zyx.insert(0,0)

    return add_cubes_to_data_from_indices(new_data, squares, cubes_len_zyx, indices, remove_zyx)