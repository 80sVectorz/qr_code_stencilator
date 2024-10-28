import cv2
import numpy as np
import random
import janky_verbosity

VALID_DIMENSIONS = np.array([21+4*i for i in range(0,40)])

def rle(in_array):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(in_array)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return (z, p, ia[i])

def read_qr_code(image_path,verbosity = 'SIMPLE'):
    print_error,print_warning,print_debug,print = janky_verbosity.create_log_functions(verbosity)

    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img,np.mean(img),255,cv2.THRESH_BINARY)[-1]

    base_img_data = None
    if int(sum(img.shape)/2) not in VALID_DIMENSIONS:
        coords = cv2.findNonZero((img < 1).astype(int)) # Find all non-zero points
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box

        img = img[y:y+h, x:x+w]

        run_lengths = []
        for i in range(img.shape[0]):
            row = img[i]==0
            row_run_lengths, start_positions, _ = rle(row)
            run_lengths += [row_run_lengths[j] for j, (rl,sp) in enumerate(zip(row_run_lengths, start_positions)) if row[sp] and rl > 1]

        rles = np.array(run_lengths).flatten()
        pixel_size_estimate = np.min(rles)

        side_length_pixels = (img.shape[0]+img.shape[1])/2 /pixel_size_estimate
        dimension_correction = np.abs(VALID_DIMENSIONS-side_length_pixels)
        side_length_pixels = VALID_DIMENSIONS[int(np.where(np.min(dimension_correction) == np.abs(VALID_DIMENSIONS-side_length_pixels))[0][0])]

        pixel_size_estimate_x = img.shape[1] / side_length_pixels
        pixel_size_estimate_y = img.shape[0] / side_length_pixels

        print_debug(f'INFO: Calculated side length (pixels): {side_length_pixels}')

        base_img_data = np.zeros((side_length_pixels,side_length_pixels))
        sample_points_x = np.array([[pixel_size_estimate_x/2+i*pixel_size_estimate_x for i in range(side_length_pixels)] for j in range(side_length_pixels)]).flatten()
        sample_points_y = np.array([[pixel_size_estimate_y/2+j*pixel_size_estimate_y for i in range(side_length_pixels)] for j in range(side_length_pixels)]).flatten()

        for px,py in zip(sample_points_x,sample_points_y):
            x = round((px-pixel_size_estimate_x/2)/pixel_size_estimate_x)
            y = round((py-pixel_size_estimate_y/2)/pixel_size_estimate_y)

            base_img_data[y,x] += img[round(py),round(px)]
    else:
        base_img_data = img
    return base_img_data

def compute_segments(img_data):
    segmented_image = img_data.copy().astype(int)
    W, H = segmented_image.shape

    x_indices = list(range(W))
    random.shuffle(x_indices)
    y_indices = list(range(H))
    random.shuffle(y_indices)

    last_pixel_count = W*H
    n_segments = 0
    n_segments_split = [0,0]
    segment_types = []
    segment_pixel_counts = []
    for i in x_indices:
        for j in y_indices:
            if segmented_image[i, j] == 0 or segmented_image[i, j] == 255:
                n_segments += 1
                n_segments_split[0 if segmented_image[i,j]==0 else 1] += 1
                segment_types.append(0 if segmented_image[i,j]==0 else 1)
                
                cv2.floodFill(segmented_image, None, (j, i), n_segments)

                new_pixel_count = np.sum(segmented_image == img_data)
                pixel_count_diff = last_pixel_count-new_pixel_count
                last_pixel_count = new_pixel_count

                segment_pixel_counts.append(pixel_count_diff)

    segmented_image -= 1
                
    return n_segments,n_segments_split,segment_types,segment_pixel_counts,segmented_image

def compute_segment_neighbors_map(segmented_image):
    segment_neighbors = [[] for i in range(int(np.max(segmented_image)+1))]

    sample_offsets = ((1,0),(-1,0),(0,1),(0,-1))

    for px in range(1,segmented_image.shape[1]-1):
        for py in range(1,segmented_image.shape[0]-1):
            segment_id = segmented_image[py,px]
            for i,j in sample_offsets:
                neighbor_id = segmented_image[py+j,px+i]
                if neighbor_id != segment_id and neighbor_id not in segment_neighbors[segment_id]:
                    segment_neighbors[segment_id].append(neighbor_id)
                    segment_neighbors[neighbor_id].append(segment_id)
    
    return segment_neighbors

def travel_to_end(segment_neighbors,all_floating_segments,visited,path_lengths,path):
    results = []
    if path[-1] in all_floating_segments and (path[-1] not in path_lengths.keys() or path_lengths[path[-1]] > len(path)):
        path_lengths[path[-1]] = len(path)
        results.append(path)
    for neighbor in segment_neighbors[path[-1]]:
        if ((path[-1],neighbor) not in visited or (neighbor in path_lengths.keys() and path_lengths[neighbor] > len(path)+1)) and neighbor not in path:
            visited.append((path[-1],neighbor))
            result = travel_to_end(segment_neighbors,all_floating_segments,visited,path_lengths,path+[neighbor])
            if result:
                results+=result
    results = [result for result in results if path_lengths[result[-1]] == len(result)]
    if len(results) > 0:
        return results
    return None

def generate_raw_stencils(raw_qr_code, verbosity='SIMPLE'):
    print_error,print_warning,print_debug,print = janky_verbosity.create_log_functions(verbosity)

    try:
        print_debug("INFO: Segmenting QR code")
        img_data = np.pad(raw_qr_code, (1,1), 'constant',constant_values=255)
        n_segments, n_segments_split, segment_types, segment_pixel_counts, segmented_image = compute_segments(img_data)

        border_segment = segmented_image[0,0]

        print_debug(f'INFO: #segments: {n_segments}')
        print_debug(f'INFO: #segments (black): {n_segments_split[0]}')
        print_debug(f'INFO: #segments (white): {n_segments_split[1]}')
    except:
        print_error(f"ERROR: Failed to compute QR code segments")
        return
    
    try:
        print_debug("INFO: Computing segment neighbors")
        segment_neighbors = compute_segment_neighbors_map(segmented_image)
    except:
        print_error(f"ERROR: Failed to compute segment neighbors")
        return

    try:
        print_debug("INFO: Computing floating segments")
        floating_segments = [[],[]]

        for segment_id in range(n_segments):
            if segment_id != border_segment:
                match(segment_types[segment_id]):
                    case 0:
                        black_neighbor_check = sum([1-segment_types[i] for i in segment_neighbors[segment_id]])
                        if black_neighbor_check == 0 and border_segment not in segment_neighbors[segment_id]:
                            floating_segments[0].append(segment_id)
                    case 1:
                        white_neighbor_check = sum([segment_types[i] for i in segment_neighbors[segment_id]])
                        if white_neighbor_check == 0 and border_segment not in segment_neighbors[segment_id]:
                            floating_segments[1].append(segment_id)

        all_floating_segments = floating_segments[0]+floating_segments[1]

    except:
        print_error(f"ERROR: Failed to compute floating segments")
        return
    
    try:
        print_debug("INFO: Mapping nested floating segments")
        visited = []
        path_lengths = {}
        tail_search_results = []
        tail_search_result = None
        last_visited_length = -1
        while len(visited) != last_visited_length:
            last_visited_length = len(visited)
            tail_search_result = travel_to_end(segment_neighbors,all_floating_segments,visited,path_lengths,[border_segment])
            if tail_search_result:
                tail_search_results += tail_search_result

        floating_segment_chains = [[chain_segment for chain_segment in chain if chain_segment in all_floating_segments] for chain in tail_search_results]

        floating_segment_chains_lut = {}

        for i,tail_chain in enumerate(tail_search_results):
            for segment_id in tail_chain:
                if segment_id in all_floating_segments:
                    floating_segment_chains_lut[segment_id] = i

        floating_segments_mask = np.zeros((segmented_image.shape[1],segmented_image.shape[0]))
        for segment_id in floating_segments[0]:
            floating_segments_mask-=(segmented_image == segment_id)
        for segment_id in floating_segments[1]:
            floating_segments_mask+=(segmented_image == segment_id)
    except:
        print_error(f"ERROR: Failed to map nested floating segments")
        return
    
    try:
        print_debug("INFO: Generating required stencils")
        target_img = img_data/255

        stencil0 = np.zeros_like(target_img)
        stencil0[1:-1,1:-1] = 1

        stencil1 = 1-target_img + np.maximum(0,floating_segments_mask)

        stencils = [stencil0,stencil1]

        floating_segment_chains_left = [x.copy() for x in floating_segment_chains]

        stencil_simulation_steps = [np.ones_like(stencil1),np.ones_like(stencil1),1-stencil1]

        last_color = 0
        while sum([len(x) for x in floating_segment_chains_left]) > 0:
            new_stencil = np.zeros_like(stencil1)

            for i,chain in enumerate(floating_segment_chains_left):
                while (
                    len(floating_segment_chains_left[i]) > 0
                    and segment_types[floating_segment_chains_left[i][0]] == min(np.sum(stencil_simulation_steps[-1]*(segmented_image==floating_segment_chains_left[i][0])),1)
                ):
                    floating_segment_chains_left[i].pop(0)

            for i,chain in enumerate(floating_segment_chains_left):
                if len(chain) > 0:
                    segments_drawn = 0

                    for j,chain_segment in enumerate(chain):
                        if ( 
                            segments_drawn == 0 
                            and segment_types[chain_segment] == min(np.sum(stencil_simulation_steps[-1]*(segmented_image==chain_segment)),1)
                        ):
                            continue
                        segments_drawn+=1
                        new_stencil+=segmented_image==chain_segment
            
            new_stencil = np.minimum(new_stencil,1)
            if np.sum(new_stencil) == 0:
                break

            stencils.append(new_stencil)
            last_color = 1-last_color
            new_simulation_step = stencil_simulation_steps[-1].copy()
            match(last_color):
                case 0:
                    new_simulation_step *= 1-new_stencil
                case 1:
                    new_simulation_step = np.minimum(new_simulation_step+new_stencil,1)
            stencil_simulation_steps.append(new_simulation_step)

    except:
        print_error(f"ERROR: Failed to generate raw stencils")
        return
    
    return stencils