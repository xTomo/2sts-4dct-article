import time

import numpy as np
from skimage.draw import disk
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import erosion, dilation
from scipy.spatial.distance import pdist


EMPTY_LEVEL = 0
FULL_LEVEL = 0.8


def get_channels_pixels(_channels_centres, _channels_radii):
    _channels_pixels = []
    for i in np.arange(len(_channels_centres)):
        _channels_pixels.append(disk(_channels_centres[i], _channels_radii[i]))
    return _channels_pixels


def create_empty_object(size, _channels_pixels, with_outer_circle=False):
    _empty_object = np.zeros((size, size))
    if with_outer_circle:
        rr, cc = disk((size // 2, size // 2), size // 2)
        _empty_object[rr, cc] = 1
    for i in np.arange(len(_channels_pixels)):
        _empty_object[_channels_pixels[i]] = 0 if with_outer_circle else 1
    return _empty_object


def create_mask_object(size, _channels_pixels):
    _mask_object = np.zeros((size, size))
    for i in np.arange(len(_channels_pixels)):
        _mask_object[_channels_pixels[i]] = 1
    return _mask_object


def create_sinogram(_object, _number_of_time_points):
    theta = np.arange(0, 180, 180 / _number_of_time_points)
    return radon(_object, theta=theta).T


def fill_forward_simulation(time_start, time_finish):

    def fill_forward(t):
        return EMPTY_LEVEL + (t - time_start) * (FULL_LEVEL - EMPTY_LEVEL) / (time_finish - time_start)

    def fill_values(t):
        return EMPTY_LEVEL if t < time_start else FULL_LEVEL if t > time_finish else fill_forward(t)

    return fill_values


def fill_backward_simulation(time_start, time_finish):

    def fill_backward(t):
        return EMPTY_LEVEL + (time_finish - t) * (FULL_LEVEL - EMPTY_LEVEL) / (time_finish - time_start)

    def fill_values(t):
        return FULL_LEVEL if t < time_start else EMPTY_LEVEL if t > time_finish else fill_backward(t)

    return fill_values


def fill_impulse_simulation(time_start, time_finish):
    time_half = (time_finish - time_start) // 2
    fill_forward = fill_forward_simulation(time_start, time_start + time_half)
    fill_backward = fill_backward_simulation(time_start + time_half, time_finish)

    def fill_values(t):
        if t < time_start or t > time_finish:
            return EMPTY_LEVEL
        elif t < time_start + time_half:
            return fill_forward(t)
        else:
            return fill_backward(t)

    return fill_values


def fill_inverse_impulse_simulation(time_start, time_finish):
    time_half = (time_finish - time_start) // 2
    fill_backward = fill_backward_simulation(time_start, time_start + time_half)
    fill_forward = fill_forward_simulation(time_start + time_half, time_finish)

    def fill_values(t):
        if t < time_start or t > time_finish:
            return FULL_LEVEL
        elif t < time_start + time_half:
            return fill_backward(t)
        else:
            return fill_forward(t)

    return fill_values


def create_experimental_objects(_number_of_time_points, size, _channels_data, _empty_object):
    _channels_pixels, _filling_times, _directions = _channels_data
    _experimental_objects = np.broadcast_to(
        _empty_object, (_number_of_time_points, size, size)
    ).copy()
    for t in np.arange(_number_of_time_points):
        img = _experimental_objects[t]
        for i in np.arange(len(_channels_pixels)):
            fill_func = lambda t: 1
            if _directions[i] == 'f':
                fill_func = fill_forward_simulation
            elif _directions[i] == 'b':
                fill_func = fill_backward_simulation
            elif _directions[i] == 'imp':
                fill_func = fill_impulse_simulation
            elif _directions[i] == '~imp':
                fill_func = fill_inverse_impulse_simulation
            img[_channels_pixels[i]] = fill_func(*_filling_times[i])(t)
        _experimental_objects[t] = img
    return _experimental_objects


def create_experimental_sinogram(_number_of_time_points, size, _experimental_objects):
    _experimental_sinogram = np.empty((_number_of_time_points, size))
    _experimental_sinograms = np.empty((_number_of_time_points, _number_of_time_points, size))
    theta = np.arange(0, 180, 180 / _number_of_time_points)
    for t in np.arange(_number_of_time_points):
        sinogram = radon(_experimental_objects[t], theta=theta)
        _experimental_sinogram[t] = sinogram.T[t]
        _experimental_sinograms[t] = sinogram.T
    return _experimental_sinogram, _experimental_sinograms


def plot_channels_area_values(_channels_pixels, _objects, _experimental_objects=None):
    number_of_channels = len(_channels_pixels)
    fig, ax = plt.subplots(1, number_of_channels, figsize=(7.5 * number_of_channels, 5))
    ax[0].set_ylabel('Intensity values')
    for i in np.arange(number_of_channels):
        yy, xx = _channels_pixels[i]
        if _experimental_objects is not None:
            ax[i].plot(np.sum(_experimental_objects[:, yy, xx], axis=1) / xx.size)
        ax[i].plot(np.sum(_objects[:, yy, xx], axis=1) / xx.size)
        ax[i].set_xlabel('time')
    plt.suptitle('Channels values', fontsize=18)
    plt.show()


def create_phantom_objects(_number_of_time_points, _number_of_channels, _thin=False, _ch_type=None, _k=2):
    size = 32 * _k + 1

    if _number_of_channels == 4:

        if _ch_type == 'sparse':
            _channels_centres = np.array([(16, 6), (6, 16), (16, 26), (26, 16)]) * _k  # for 32_4_sparse
        elif _ch_type == 'tight':
            _channels_centres = np.array([(16, 12), (8, 16), (16, 20), (24, 16)]) * _k  # for 32_4_tight
        else:
            _channels_centres = np.array([(16, 8), (8, 16), (16, 24), (24, 16)]) * _k  # for 32_4_regular

        _filling_times = np.array([(40, 140), (40, 140), (20, 100), (80, 160)]) * _number_of_time_points / 180
        _directions = ['f', 'b', 'imp', '~imp']

    elif _number_of_channels == 8:

        _channels_centres = np.array(
            [(14, 15), (18, 9), (5, 13), (20, 26), (10, 6), (26, 20), (10, 23), (25, 8)]) * _k  # for 32_8

        _filling_times = np.array(
            [(40, 140), (40, 140), (20, 100), (120, 160), (80, 100), (20, 60), (40, 60), (20, 160)]
        ) * _number_of_time_points / 180
        _directions = ['f', 'b', 'imp', '~imp', 'f', 'b', 'imp', '~imp']

    else:
        raise ValueError('number_of_channels value should be 4 or 8')

    _channels_radius = 2 if (_thin or _number_of_channels == 8) else 4
    _channels_radii = np.ones(_channels_centres.shape[0]) * _channels_radius * _k

    _channels_pixels = get_channels_pixels(_channels_centres, _channels_radii)

    _empty_object = create_empty_object(size, _channels_pixels, with_outer_circle=True)
    _mask_object = create_mask_object(size, _channels_pixels)

    _empty_sinogram = create_sinogram(_empty_object, _number_of_time_points)
    _mask_sinogram = create_sinogram(_mask_object, _number_of_time_points)

    _channels_data = (_channels_pixels, _filling_times, _directions)

    _experimental_objects = create_experimental_objects(_number_of_time_points, size, _channels_data, _empty_object)
    _experimental_sinogram, _experimental_sinograms = create_experimental_sinogram(_number_of_time_points, size,
                                                                                   _experimental_objects)

    return (
        _empty_object,
        _mask_object,

        _empty_sinogram,
        _mask_sinogram,

        _channels_data,

        _experimental_objects,

        _experimental_sinogram,
        _experimental_sinograms,
    )


def normalize_sinograms(_sinograms, _experimental_sinogram):
    output_sinograms = np.empty(_sinograms.shape)
    for i, e_sino_row in enumerate(_experimental_sinogram):
        invariant = np.sum(e_sino_row)
        output_sinograms[i] = np.array([invariant * row / np.sum(row) for row in _sinograms[i]])
    return output_sinograms


def insert_exp_values(_sinograms, _experimental_sinogram):
    output_sinograms = _sinograms.copy()
    for i, sino_row in enumerate(_experimental_sinogram):
        output_sinograms[i, i] = sino_row
    return output_sinograms


def create_initial_sinograms(_empty_sinogram, _experimental_sinogram, _number_of_time_points):
    sino_array_shape = (_number_of_time_points, *_empty_sinogram.shape)
    empty_sinograms = np.tile(_empty_sinogram.flatten(), _number_of_time_points).reshape(sino_array_shape)
    initial_sinograms = insert_exp_values(empty_sinograms, _experimental_sinogram)
    return normalize_sinograms(initial_sinograms, _experimental_sinogram)


def create_sino(_object, _number_of_time_points):
    theta = np.arange(0, 180, 180/_number_of_time_points)
    return radon(_object, theta=theta).T


def l2_norm(obj1, obj2):
    return np.sqrt(np.sum(np.square(obj1 - obj2)))


def rrmse(obj1, gt_obj):
    return l2_norm(obj1, gt_obj) / np.sqrt(np.sum(np.square(gt_obj)))


def calc_intermediate_objects(_sinograms, _mask_object, _number_of_time_points):
    inter_objects = np.empty((_number_of_time_points, *_mask_object.shape))
    theta = np.arange(0, 180, 180/_number_of_time_points)
    for i in np.arange(_number_of_time_points):
        inter_objects[i] = iradon(_sinograms[i].T, theta=theta)
    return inter_objects


def norm_intermediate_objects(inter_objects, _mask_object, _channels_pixels=None):
    for i, _ in enumerate(inter_objects):
        obj_sum_before = inter_objects[i].sum()
        inter_objects[i] = inter_objects[i] * _mask_object
        obj_sum_after = np.sum(inter_objects[i])
        inter_objects[i] *= obj_sum_before/obj_sum_after
        if _channels_pixels is not None:
            for _, _ch_pixels in enumerate(_channels_pixels):
                yy, xx = _ch_pixels
                mean_v = inter_objects[i, yy, xx].mean()
                level = EMPTY_LEVEL if (mean_v < EMPTY_LEVEL) else FULL_LEVEL if (mean_v > FULL_LEVEL) else mean_v
                inter_objects[i, yy, xx] = level
    return inter_objects


def get_iteration_sinograms(_objects, _number_of_time_points):
    iteration_sinograms = np.empty((_number_of_time_points, _number_of_time_points, _objects.shape[-1]))
    theta = np.arange(0, 180, 180/_number_of_time_points)
    for i in np.arange(_number_of_time_points):
        sinogram = radon(_objects[i], theta=theta)
        iteration_sinograms[i] = sinogram.T
    return iteration_sinograms


def construct_exp_sinogram(_sinograms):
    exp_sinogram = np.empty(_sinograms[0].shape)
    for i in np.arange(_sinograms.shape[0]):
        exp_sinogram[i] = _sinograms[i, i, :]
    return exp_sinogram


def sort_objects_min(_objects):
    s_objects = np.empty(_objects.shape)
    for (i, _object) in enumerate(_objects):
        if i == _objects.shape[0] - 1:
            s_objects[i] = _object
        else:
            s_objects[i] = np.minimum(_object, _objects[i+1])
    return s_objects


def sort_objects_median(_objects, _number_of_time_points):
    s_objects = np.copy(_objects)
    for i in np.arange(_number_of_time_points):
        if i == 0:
            s_objects[i] = np.median([_objects[i], _objects[i+1]], axis=0)
        elif i == _number_of_time_points - 1:
            s_objects[i] = np.median([_objects[i], _objects[i-1]], axis=0)
        else:
            s_objects[i] = np.median([_objects[i-1], _objects[i], _objects[i+1]], axis=0)
    return s_objects


def gaussian_filter_of_objects_in_time(_objects, sigma=1):
    return gaussian_filter(_objects, sigma=sigma, axes=(0,))


def extrapolate_exp_sino_line(i, _number_of_time_points, s_sino, power=None):
    for j in range(i-1, -1, -1):
        k = 1 / (i + 1 - j)
        k = np.power(k, power) if power is not None else k
        s_sino[j] = s_sino[j] + k * (s_sino[j+1] - s_sino[j])
    for j in range(i+1, _number_of_time_points, 1):
        k = 1 / (j - i + 1)
        k = np.power(k, power) if power is not None else k
        s_sino[j] = s_sino[j] + k * (s_sino[j-1] - s_sino[j])
    return s_sino


def extrapolate_exp_sino_data(_number_of_time_points, s_sinograms, power=None):
    for i in range(_number_of_time_points):
        s_sino = np.copy(s_sinograms[:, i, :])
        s_sinograms[:, i, :] = extrapolate_exp_sino_line(i, _number_of_time_points, s_sino, power=power)
    return s_sinograms


def calc_2sts_4dct(_empty_obj,
                   _mask_object,
                   _empty_sino,
                   _mask_sino,
                   _channels_data,
                   _gt_exp_objects,
                   _gt_exp_sino,
                   _number_of_time_points,
                   iteration_stop_diff=0.01,
                   alg_type='new',
                   show_info=True):
    start = time.time()

    times = np.floor(np.array([10, 60, 120, 170]) * _number_of_time_points / 180).astype(int)

    _channels_pixels, _, _ = _channels_data

    max_number_of_iterations = 1001

    full_objects = np.zeros(_gt_exp_objects.shape)
    full_e_sino = np.zeros(_gt_exp_sino.shape)
    full_sino_rrmse = []
    full_obj_rrmse = []
    full_sino_rrmse_threshold_reached = False

    _exp_sino = _gt_exp_sino - _empty_sino

    i_objects = []
    i_sinograms = create_initial_sinograms(_mask_sino, _exp_sino, _number_of_time_points)
    if show_info:
        fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 5))
        for index, axis in enumerate(axes):
            axis.imshow(i_sinograms[times[index]], aspect='auto')
        plt.show()
        print(f'i_sinograms shape: {i_sinograms.shape}')

    final_number_of_iterations = None

    for i in np.arange(max_number_of_iterations):

        if i > 0:
            i_sinograms = insert_exp_values(i_sinograms, _exp_sino)
            i_sinograms = normalize_sinograms(i_sinograms, _exp_sino)

        if alg_type == 'new':
            power = 0.5
            i_sinograms = extrapolate_exp_sino_data(_number_of_time_points, i_sinograms, power=power)
            i_sinograms = normalize_sinograms(i_sinograms, _exp_sino)

        i_objects = calc_intermediate_objects(i_sinograms, _mask_object, _number_of_time_points)
        i_objects = norm_intermediate_objects(i_objects, _mask_object, _channels_pixels)

        if alg_type == 'old_min_sort':
            i_objects = sort_objects_min(i_objects)

        if alg_type == 'old_no_sort':
            pass

        if alg_type == 'old_median':
            i_objects = sort_objects_median(i_objects, _number_of_time_points)

        if alg_type == 'old_gauss_filter':
            i_objects = gaussian_filter_of_objects_in_time(i_objects)

        i_sinograms = get_iteration_sinograms(i_objects, _number_of_time_points)

        full_objects = i_objects + _empty_obj
        full_obj_rrmse.append(rrmse(full_objects, _gt_exp_objects))

        full_sinograms = i_sinograms + _empty_sino
        full_e_sino = construct_exp_sinogram(full_sinograms)
        full_sino_rrmse.append(rrmse(full_e_sino, _gt_exp_sino))

        if i % 10 == 0:
            if show_info:
                print(f'iteration: {i},\t sino rsme: {full_sino_rrmse[i]},\t objects rsme: {full_obj_rrmse[i]}')

        if (i % 50 == 0 or i == 10) and show_info:
            plot_channels_area_values(_channels_pixels, full_objects, _gt_exp_objects)

        rrmse_diff = None
        if not full_sino_rrmse_threshold_reached and i > 0:
            rrmse_diff = full_sino_rrmse[i - 1] - full_sino_rrmse[i]
            if (rrmse_diff is not None) and (rrmse_diff < full_sino_rrmse[i] * iteration_stop_diff):
                full_sino_rrmse_threshold_reached = True

        if full_sino_rrmse_threshold_reached:
            final_number_of_iterations = i
            if show_info:
                print(f'\n')
                print(f'full_sino_rrmse diff reached {iteration_stop_diff} at {i} iteration')
                print(f'{full_sino_rrmse[i - 1]}, {full_sino_rrmse[i]}, diff: {rrmse_diff}')
                print(f'\n')
                print(f'loop break at {i} iteration due to full_sino_rrmse_threshold_reached')
            break

    dyn_obj = _gt_exp_objects * _mask_object
    dyn_obj_rrmse = rrmse(i_objects, dyn_obj)
    dyn_sino_rrmse = rrmse(construct_exp_sinogram(i_sinograms), _exp_sino)

    end = time.time()
    duration = end - start
    final_number_of_iterations = final_number_of_iterations or max_number_of_iterations

    if show_info:

        plot_channels_area_values(_channels_pixels, full_objects, _gt_exp_objects)

        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        ax[0].plot(full_sino_rrmse)
        ax[0].set_title('Sinogram RRMSE vs iteration number')
        ax[1].plot(full_obj_rrmse)
        ax[1].set_title('Object RRMSE vs iteration number')
        plt.show()

        fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 5))
        vmax = np.max(_gt_exp_objects[times])
        vmin = np.min(_gt_exp_objects[times])
        for i in range(len(times)):
            im = axes[i].imshow(_gt_exp_objects[times[i]], vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axes[i])
        fig.suptitle('ground truth experimental objects')
        plt.show()

        fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 5))
        vmax = np.max(full_objects[times])
        vmin = np.min(full_objects[times])
        for i in range(len(times)):
            im = axes[i].imshow(full_objects[times[i]], vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axes[i])
        fig.suptitle('reconstructed experimental objects')
        plt.show()

    print(f'{final_number_of_iterations} iterations takes {duration} seconds')
    print(f'final sino rrmse: {full_sino_rrmse[-1]},\t objects rrmse: {full_obj_rrmse[-1]}')
    print(f'final dyn sino rrmse: {dyn_sino_rrmse},\t dyn objects rrmse: {dyn_obj_rrmse}')
    print('\n')

    return {
        'iterations': final_number_of_iterations,
        'duration': duration,
        'recon_objects': full_objects,
        'recon_e_sino': full_e_sino,
        'sino_rrmse': full_sino_rrmse,
        'obj_rrmse': full_obj_rrmse,
        'dyn_sino_rrmse': dyn_sino_rrmse,
        'dyn_obj_rrmse': dyn_obj_rrmse,
    }


def create_blank_object(size):
    _blank_object = np.zeros((size, size))
    yy, xx = disk((size // 2, size // 2), size // 2)
    _blank_object[yy, xx] = 1
    return _blank_object


def generate_random_channels(_blank_object, _ch_radius, _number_of_ch, gap=2, _counter=0):
    _empty_object = _blank_object.copy()
    _mask_object = np.zeros(_blank_object.shape)
    _channels_centers = []
    _channels_pixels = []

    _radius = np.ceil(_ch_radius).astype(int)
    _channel_mask = np.zeros((_radius * 2 - 1 + gap, _radius * 2 - 1 + gap))
    yy, xx = disk((_radius - 1, _radius - 1), _ch_radius + gap)
    _channel_mask[yy, xx] = 1

    for _ in np.arange(_number_of_ch):

        _eroded_object = erosion(_empty_object, footprint=_channel_mask)

        _available_coordinates = np.stack(np.nonzero(_eroded_object))
        if _available_coordinates.shape[1] > 0:
            _rnd_coord = np.random.choice(_available_coordinates.shape[1])
            _new_center = _available_coordinates[:, _rnd_coord]
            yy, xx = disk(tuple(_new_center), _ch_radius)
            _empty_object[yy, xx] = 0
            _mask_object[yy, xx] = 1
            _channels_centers.append(_new_center)
            _channels_pixels.append((yy, xx))
        elif _counter == 10:
            print(f'check input values, can not generate {_number_of_ch} channels during {_counter} attempts')
            break
        else:
            return generate_random_channels(_blank_object, _ch_radius, _number_of_ch, gap=gap, _counter=_counter + 1)

    _channels_distance = pdist(np.array(_channels_centers))
    _channels_stats = {
        'distance': _channels_distance,
        'mean': np.mean(_channels_distance),
        'std': np.std(_channels_distance),
        'median': np.median(_channels_distance),
    }

    return _empty_object, _mask_object, _channels_pixels, _channels_stats


def generate_random_channels_with_exact_gap(_blank_object, _ch_radius, _number_of_ch, gap=2, _counter=0,
                                            show_images=False):
    _empty_object = _blank_object.copy()
    _mask_object = np.zeros(_blank_object.shape)
    _channels_centers = []
    _channels_pixels = []

    _radius = np.ceil(_ch_radius + 1).astype(int)

    _ch_mask_size = _radius * 2 + 1
    _ch_mask_shape = (_ch_mask_size, _ch_mask_size)
    _channel_mask = np.zeros(_ch_mask_shape)
    yy, xx = disk((_ch_mask_size // 2, _ch_mask_size // 2), _ch_radius + 1)
    _channel_mask[yy, xx] = 1

    _ch_mask_size = (_radius + gap) * 2 + 1
    _ch_mask_shape = (_ch_mask_size, _ch_mask_size)
    _channel_mask_with_gap_small = np.zeros(_ch_mask_shape)
    yy, xx = disk((_ch_mask_size // 2, _ch_mask_size // 2), _ch_radius + gap)
    _channel_mask_with_gap_small[yy, xx] = 1

    _ch_mask_size = (_radius + gap + 1) * 2 + 1
    _ch_mask_shape = (_ch_mask_size, _ch_mask_size)
    _channel_mask_with_gap_big = np.zeros(_ch_mask_shape)
    yy, xx = disk((_ch_mask_size // 2, _ch_mask_size // 2), _ch_radius + gap + 1)
    _channel_mask_with_gap_big[yy, xx] = 1

    _base_eroded_object = erosion(_empty_object, footprint=_channel_mask)

    if show_images:
        fig, ax = plt.subplots(1, 5, figsize=(35, 5))
        ax[0].imshow(_channel_mask)
        ax[1].imshow(_channel_mask_with_gap_small)
        ax[2].imshow(_channel_mask_with_gap_big)
        ax[3].imshow(_base_eroded_object)
        ax[4].imshow(_empty_object * (1 - _base_eroded_object))
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[3].grid()
        ax[4].grid()

    for i in np.arange(_number_of_ch):

        _dilated_mask_small = dilation(_mask_object, footprint=_channel_mask_with_gap_small)
        _dilated_mask_big = dilation(_mask_object, footprint=_channel_mask_with_gap_big)

        _eroded_object = _base_eroded_object * (1 - _dilated_mask_small)

        _available_centers = _eroded_object if i == 0 else _eroded_object * _dilated_mask_big

        if show_images:
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(_dilated_mask_small)
            ax[1].imshow(_dilated_mask_big)
            ax[2].imshow(_eroded_object)
            ax[3].imshow(_available_centers)

        _available_coordinates = np.stack(np.nonzero(_available_centers))

        if _available_coordinates.shape[1] > 0:
            _rnd_coord = np.random.choice(_available_coordinates.shape[1])
            _new_center = _available_coordinates[:, _rnd_coord]
            yy, xx = disk(tuple(_new_center), _ch_radius)
            _empty_object[yy, xx] = 0
            _mask_object[yy, xx] = 1
            _channels_centers.append(_new_center)
            _channels_pixels.append((yy, xx))
            if show_images:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(_empty_object)
                ax[1].imshow(_mask_object)
        elif _counter == 10:
            print(f'check input values, can not generate {_number_of_ch} channels during {_counter} attempts')
            break
        else:
            return generate_random_channels_with_exact_gap(_blank_object,
                                                           _ch_radius,
                                                           _number_of_ch,
                                                           gap=gap,
                                                           show_images=False,
                                                           _counter=_counter + 1)

    _channels_distance = pdist(np.array(_channels_centers))
    _channels_stats = {
        'distance': _channels_distance,
        'mean': np.mean(_channels_distance),
        'std': np.std(_channels_distance),
        'median': np.median(_channels_distance),
    }

    return _empty_object, _mask_object, _channels_pixels, _channels_stats


def create_random_phantom_objects(_number_of_time_points, _empty_object, _mask_object, _channels_pixels, _size):
    number_of_channels = len(_channels_pixels)

    _directions_base = ['f', 'b', 'imp', '~imp']
    idx = np.random.choice(len(_directions_base), number_of_channels)
    _directions = [_directions_base[i] for i in idx]
    # print(_directions)

    _filling_times = np.array(
        [tuple(np.sort(np.random.choice(180 - 2, 2, replace=False) + 1)) for _ in np.arange(number_of_channels)]
    ) * _number_of_time_points / 180
    # print(_filling_times)

    _empty_sinogram = create_sinogram(_empty_object, _number_of_time_points)
    _mask_sinogram = create_sinogram(_mask_object, _number_of_time_points)

    _channels_data = (_channels_pixels, _filling_times, _directions)

    _experimental_objects = create_experimental_objects(_number_of_time_points, _size, _channels_data, _empty_object)
    _experimental_sinogram, _experimental_sinograms = create_experimental_sinogram(_number_of_time_points, _size,
                                                                                   _experimental_objects)

    return (
        _empty_sinogram,
        _mask_sinogram,

        _channels_data,

        _experimental_objects,

        _experimental_sinogram,
        _experimental_sinograms,
    )
