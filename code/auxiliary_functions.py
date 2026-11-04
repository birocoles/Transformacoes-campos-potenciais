import numpy as np
from gravmag import utils


def signed_center_mass(data, x, y):
    
    # Define masks for positive and negative data values
    positive = (data > 0.)
    negative = (data < 0.)
    
    # Compute the center of mass of the positive data
    x0_positive = (
        np.sum(data[positive]*x[positive]) / 
        np.sum(data[positive])
    )
    y0_positive = (
        np.sum(data[positive]*y[positive]) / 
        np.sum(data[positive])
    )
    
    # Compute the center of mass of the negative data
    x0_negative = (
        np.sum(data[negative]*x[negative]) / 
        np.sum(data[negative])
    )
    y0_negative = (
        np.sum(data[negative]*y[negative]) / 
        np.sum(data[negative])
    )
    
    # Compute the signed center of mass
    x0 = (x0_positive + x0_negative) / 2
    y0 = (y0_positive + y0_negative) / 2

    return x0, y0


def fill_nan_with_neighbor_mean_numpy(arr, max_iter=10, verbose=True):
    """
    Fill np.nan values in a 2D numpy array by the mean of adjacent neighbors
    (including diagonals). Repeat until no NaNs remain or max_iter is reached.
    NumPy-only implementation with verbose logging.
    
    Parameters:
    - arr: 2D numpy array with float values and np.nan.
    - max_iter: Maximum number of iterations.
    - verbose: If True, prints detailed logs each iteration.
    
    Returns:
    - Filled 2D numpy array.
    """
    filled = arr.copy()
    rows, cols = filled.shape

    # Define shifts: 8 directions (diagonal + cardinal)
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

    for iteration in range(max_iter):
        nan_mask = np.isnan(filled)
        if not np.any(nan_mask):
            if verbose:
                print(f"[Iteration {iteration}] All NaNs filled. Stopping.")
            break

        neighbor_sum = np.zeros_like(filled)
        neighbor_count = np.zeros_like(filled, dtype=int)

        for dx, dy in shifts:
            shifted = np.full_like(filled, np.nan)
            
            # Compute valid slicing indices for shifting
            x_src_start = max(0, -dx)
            x_src_end   = rows - max(0, dx)
            x_dst_start = max(0, dx)
            x_dst_end   = rows - max(0, -dx)

            y_src_start = max(0, -dy)
            y_src_end   = cols - max(0, dy)
            y_dst_start = max(0, dy)
            y_dst_end   = cols - max(0, -dy)
            
            # Copy shifted values
            shifted[x_dst_start:x_dst_end, y_dst_start:y_dst_end] = \
                filled[x_src_start:x_src_end, y_src_start:y_src_end]

            # Add to sum where not NaN
            valid_mask = ~np.isnan(shifted)
            neighbor_sum += np.where(valid_mask, shifted, 0)
            neighbor_count += valid_mask.astype(int)

        # Compute mean safely
        neighbor_mean = np.divide(
            neighbor_sum, 
            neighbor_count, 
            out=np.zeros_like(neighbor_sum), 
            where=neighbor_count != 0
        )

        # New values to be filled
        new_values = neighbor_mean[nan_mask]

        # Logging
        if verbose:
            n_to_fill = np.sum(nan_mask)
            mean_new_values = np.mean(new_values) if new_values.size > 0 else np.nan
            print(f"[Iteration {iteration}] NaNs to fill: {n_to_fill}, "
                  f"Mean of new fill values: {mean_new_values:.4f}")

        # Check if something changed
        prev_values = filled[nan_mask]
        if np.allclose(new_values, prev_values, equal_nan=True):
            if verbose:
                print(f"[Iteration {iteration}] No change detected. Stopping.")
            break

        filled[nan_mask] = new_values

    return filled


def data_statistics(data, name):
    print(f'{name}:')
    print(f"min    = {np.min(data):>7.2f} m")
    print(f"mean   = {np.mean(data):>7.2f} m")
    print(f"stdev  = {np.std(data):>7.2f} m")
    print(f"median = {np.median(data):>7.2f} m")
    print(f"max    = {np.max(data):>7.2f} m")


def declination_from_bz(coordinates, bz, mask=None):
    if mask is None:
        x0_bz, y0_bz = signed_center_mass(
            data=bz, 
            x=coordinates['x'], 
            y=coordinates['y']
        )
        Mx_bz = float(-np.sum(bz*(coordinates['x'] - x0_bz)))
        My_bz = float(-np.sum(bz*(coordinates['y'] - y0_bz)))
    else:
        x0_bz, y0_bz = signed_center_mass(
            data=bz[mask], 
            x=coordinates['x'][mask], 
            y=coordinates['y'][mask]
        )
        Mx_bz = float(-np.sum(bz[mask]*(coordinates['x'][mask] - x0_bz)))
        My_bz = float(-np.sum(bz[mask]*(coordinates['y'][mask] - y0_bz)))
    estimated_declination = float(np.rad2deg(utils.safe_atan2(My_bz, Mx_bz)))
    if estimated_declination > 90:
        estimated_declination -= 180
    if estimated_declination < -90:
        estimated_declination += 180
    return estimated_declination, Mx_bz, My_bz


def inclination_from_V(Mx, My, V, mask=None):
    Mh = np.sqrt(Mx**2 + My**2)
    if mask is None:
        #Mz = float(-np.sum(V))*1e3
        Mz = float(np.sum(V))
    else:
        #Mz = float(-np.sum(V[mask]))*1e3
        Mz = float(np.sum(V[mask]))
    estimated_inclination = float(np.rad2deg(utils.safe_atan2(Mz,Mh)))
    # if estimated_inclination > 90:
    #     estimated_inclination -= 180
    # if estimated_inclination < -90:
    #     estimated_inclination += 180
    return estimated_inclination, Mz


def inclination_from_bx(coordinates, Mx, My, bx, mask=None):
    Mh = np.sqrt(Mx**2 + My**2)
    if mask is None:
        x0, y0 = signed_center_mass(
            data=bx, 
            x=coordinates['x'],
            y=coordinates['y']
        )
        Mz = float(-np.sum(bx*(coordinates['x'] - x0)))
    else:
        x0, y0 = signed_center_mass(
            data=bx[mask], 
            x=coordinates['x'][mask],
            y=coordinates['y'][mask]
        )
        Mz = float(-np.sum(bx[mask]*(coordinates['x'][mask] - x0)))
    estimated_inclination = float(np.rad2deg(utils.safe_atan2(Mz,Mh)))
    # if estimated_inclination > 90:
    #     estimated_inclination -= 180
    # if estimated_inclination < -90:
    #     estimated_inclination += 180
    return estimated_inclination, Mz

def inclination_from_by(coordinates, Mx, My, by, mask=None):
    Mh = np.sqrt(Mx**2 + My**2)
    if mask is None:
        x0, y0 = signed_center_mass(
            data=by, 
            x=coordinates['x'],
            y=coordinates['y']
        )
        Mz = float(-np.sum(by[mask]*(coordinates['y'][mask] - y0)))
    else:
        x0, y0 = signed_center_mass(
            data=by[mask], 
            x=coordinates['x'][mask],
            y=coordinates['y'][mask]
        )
        Mz = float(-np.sum(by*(coordinates['y'] - y0)))
    estimated_inclination = float(np.rad2deg(utils.safe_atan2(Mz,Mh)))
    # if estimated_inclination > 90:
    #     estimated_inclination -= 180
    # if estimated_inclination < -90:
    #     estimated_inclination += 180
    return estimated_inclination, Mz