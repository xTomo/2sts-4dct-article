# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% id="WHzwjyZOTCsp"
import _fn
import matplotlib.pyplot as plt
import numpy as np



# %% [markdown] id="nPq2GbYxeXbT"
# # **Phantom objects generation**

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="gMGZr0nvbV8C" outputId="c50ed7c4-f6ed-4c59-e180-0563c09f25d8"
number_of_time_points = 90

phantom_data = _fn.create_phantom_objects(number_of_time_points, 4)
# phantom_data = _fn.create_phantom_objects(number_of_time_points, 4, _ch_type='sparse')
# phantom_data = _fn.create_phantom_objects(number_of_time_points, 4, _ch_type='tight')
# phantom_data = _fn.create_phantom_objects(number_of_time_points, 8)

(empty_object,
 mask_object,

 empty_sinogram,
 mask_sinogram,

 channels_data,

 experimental_objects,

 experimental_sinogram,
 experimental_sinograms) = phantom_data

_channels_pixels, _, _ = channels_data

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(empty_object)
axes[0].set_title('Empty object')
axes[1].imshow(mask_object)
axes[1].set_title('Mask')
axes[2].imshow(empty_sinogram, aspect='auto')
axes[2].set_title('Empty object sinogram')
axes[3].imshow(mask_sinogram, aspect='auto')
axes[3].set_title('Mask sinogram')
plt.show()

_fn.plot_channels_area_values(_channels_pixels, experimental_objects)

times = np.floor(np.array([10, 60, 120, 170]) * number_of_time_points / 180).astype(int)
vmax = np.max(empty_sinogram)

dynamic_region_sinogram = experimental_sinogram - empty_sinogram

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(empty_sinogram, vmax=vmax, aspect='auto')
axes[0].set_title('Empty object sinogram')
axes[1].imshow(experimental_sinogram, vmax=vmax, aspect='auto')
for time_point in times:
    axes[1].axhline(time_point, color='lightgray')
axes[1].set_title('Experimental sinogram')
axes[2].imshow(mask_sinogram, vmax=vmax, aspect='auto')
axes[2].set_title('Mask sinogram')
axes[3].imshow(dynamic_region_sinogram, vmax=vmax, aspect='auto')
axes[3].set_title('Dynamic region sinogram')
plt.show()

fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
for index, axis in enumerate(axes):
    axis.imshow(experimental_objects[times[index]], vmin=0, vmax=1)
    axis.set_title(f'Object at time step #{times[index]}')
plt.show()

print(f'empty_object shape: {empty_object.shape}')
print(f'empty_sinogram shape: {empty_sinogram.shape}')
print(f'experimental_objects shape: {experimental_objects.shape}')
print(f'experimental_sinogram shape: {experimental_sinogram.shape}')
print(f'experimental_sinograms shape: {experimental_sinograms.shape}')

# %% [markdown] id="w73ulfrhemjZ"
# # **Algorythm implementation**

# %% [markdown]
# ### 1. Create initial sinograms

# %%
sino_array_shape = (number_of_time_points, *mask_sinogram.shape)
initial_sinograms = np.tile(mask_sinogram.flatten(), number_of_time_points).reshape(sino_array_shape)

fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 5))
plt.suptitle('Initial sinograms')
for index, axis in enumerate(axes):
    axis.imshow(initial_sinograms[times[index]], aspect='auto')
    axis.set_title(f'at time step #{times[index]}')
plt.show()

print(initial_sinograms.shape)

# %% [markdown]
# ### 2. Insert experimental values into sinograms

# %% colab={"base_uri": "https://localhost:8080/", "height": 442} id="1hgLTfB6UyNA" outputId="99f78183-a479-47f4-9278-183861b46dc0"
i_sinograms = _fn.insert_exp_values(initial_sinograms, dynamic_region_sinogram)
i_sinograms = _fn.normalize_sinograms(i_sinograms, dynamic_region_sinogram)

fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 5))
plt.suptitle('Intermediate sinograms')
for index, axis in enumerate(axes):
    axis.imshow(i_sinograms[times[index]], aspect='auto')
    axis.set_title(f'at time step #{times[index]}')
plt.show()

print(i_sinograms.shape)

# %% [markdown]
# ### 3. Make exponential smoothing at projections domain

# %%
angle_to_show = 10
column_to_show = 15

i_sino = np.copy(i_sinograms[:, angle_to_show, :])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.suptitle(f'Set of projections for angle {angle_to_show}')

axes[0].imshow(i_sino, aspect='auto')
axes[0].axvline(column_to_show, linewidth=4, color=u'#dddddd50')
axes[0].set_title('before processing')
axes[0].set_ylabel('time step')

power = 0.5
i_sinograms = _fn.extrapolate_exp_sino_data(number_of_time_points, i_sinograms, power=power)

axes[1].imshow(i_sinograms[:, angle_to_show, :], aspect='auto')
axes[1].axvline(column_to_show, linewidth=4, color=u'#dddddd50')
axes[1].set_title('after processing')
axes[1].set_ylabel('time step')

plt.show()

fig, axes = plt.subplots(1, 1, figsize=(10, 5))
plt.suptitle(f'Projections values profile at column {column_to_show}')
axes.plot(i_sino[:, column_to_show], c='red', label='before processing')
axes.plot(i_sinograms[:, angle_to_show, column_to_show], c='blue', label='after processing')
axes.set_xlabel('time step')
plt.grid()
plt.legend()
plt.show()

# %% [markdown]
# ### 4. Calculate intermediate objects

# %% id="pOCNc4yZgPMp"
i_objects = _fn.calc_intermediate_objects(i_sinograms, mask_object, number_of_time_points)
print(i_objects.shape)

fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
plt.suptitle('Intermediate objects')
for index, axis in enumerate(axes):
    axis.imshow(i_objects[times[index]], vmin=0, vmax=1)
    axis.set_title(f'at time step #{times[index]}')
plt.show()


# %% [markdown]
# ### 5-6. Apply mask to objects and equalize channels values

# %%
i_objects = _fn.norm_intermediate_objects(i_objects, mask_object, _channels_pixels)
print(i_objects.shape)

fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
plt.suptitle('Intermediate objects')
for index, axis in enumerate(axes):
    axis.imshow(i_objects[times[index]], vmin=0, vmax=1)
    axis.set_title(f'at time step #{times[index]}')
plt.show()


# %% [markdown]
# ### 7. Calculate intermediate sinograms

# %%
i_sinograms = _fn.get_iteration_sinograms(i_objects, number_of_time_points)
print(i_sinograms.shape)

vmax = np.max(i_sinograms)
vmin = np.min(i_sinograms)

fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 5))
plt.suptitle('Intermediate sinograms')
for index, axis in enumerate(axes):
    axis.imshow(i_sinograms[times[index]], aspect='auto', vmax=vmax, vmin=vmin)
    axis.set_title(f'at time step #{times[index]}')
plt.show()

print(i_sinograms.shape)

# %% [markdown]
# ### 8. Check RRMSE and go to Step 2 or exit

# %% [markdown]
# # **Test different algorithms with different objects**

# %% [markdown]
# ### New algorithm with 4-channel "regular" object, iteration_stop_diff = 0.001

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="uMxyVAMRS4Yp" outputId="47c159c3-2831-487c-d33f-9194c286cc0c"
phantom_data = _fn.create_phantom_objects(number_of_time_points, 4)

_ = _fn.calc_2sts_4dct(*phantom_data[:-1], number_of_time_points, alg_type='new', show_info=True, iteration_stop_diff = 0.001)


# %% [markdown]
# ### Old algorithm with "min-sort", 4-channel "regular" object, iteration_stop_diff = 0.001

# %%
_ = _fn.calc_2sts_4dct(*phantom_data[:-1], number_of_time_points, alg_type='old_min_sort', show_info=False, iteration_stop_diff = 0.001)

# %% [markdown]
# ### Old algorithm with "no-sort", 4-channel "regular" object, iteration_stop_diff = 0.001

# %%
_ = _fn.calc_2sts_4dct(*phantom_data[:-1], number_of_time_points, alg_type='old_no_sort', show_info=False, iteration_stop_diff = 0.001)

# %% [markdown]
# ### Old algorithm with median filter, 4-channel "regular" object, iteration_stop_diff = 0.001

# %%
_ = _fn.calc_2sts_4dct(*phantom_data[:-1], number_of_time_points, alg_type='old_median', show_info=False, iteration_stop_diff = 0.001)

# %% [markdown]
# ### Old algorithm with gaussian filter, 4-channel "regular" object, iteration_stop_diff = 0.001

# %%
_ = _fn.calc_2sts_4dct(*phantom_data[:-1], number_of_time_points, alg_type='old_gauss_filter', show_info=False, iteration_stop_diff = 0.001)

# %% [markdown]
# ### New algorithm with 4-channel "sparse" object, iteration_stop_diff = 0.001

# %%
phantom_data = _fn.create_phantom_objects(number_of_time_points, 4, _ch_type='sparse')

_ = _fn.calc_2sts_4dct(*phantom_data[:-1], number_of_time_points, alg_type='new', show_info=False, iteration_stop_diff = 0.001)

# %% [markdown]
# ### New algorithm with 4-channel "tight" object, iteration_stop_diff = 0.001

# %%
phantom_data = _fn.create_phantom_objects(number_of_time_points, 4, _ch_type='tight')

_ = _fn.calc_2sts_4dct(*phantom_data[:-1], number_of_time_points, alg_type='new', show_info=False, iteration_stop_diff = 0.001)

# %% [markdown]
# ### New algorithm with 8-channel object, iteration_stop_diff = 0.001

# %%
phantom_data = _fn.create_phantom_objects(number_of_time_points, 8)

_ = _fn.calc_2sts_4dct(*phantom_data[:-1], number_of_time_points, alg_type='new', show_info=False, iteration_stop_diff = 0.001)

# %% [markdown]
# # **Run algorithm on bulk of objects with random positions of channels**

# %% [markdown]
# ### 1000 number of tests may be too long, so you can set a less value, 100 or 10

# %% jupyter={"outputs_hidden": true}
# --- 65 pix ---

# k = 2

# ch_radius = 4
# number_of_ch = 8
# max_gap = 10

# ch_radius = 2
# number_of_ch = 16
# max_gap = 10

# ch_radius = 1
# number_of_ch = 32
# max_gap = 8


# --- 129 pix ---

k = 4

ch_radius = 2
number_of_ch = 32
max_gap = 15

# ch_radius = 4
# number_of_ch = 16
# max_gap = 20

# ch_radius = 8
# number_of_ch = 8
# max_gap = 22

size = 32 * k + 1

blank_object = _fn.create_blank_object(size)

n_of_iterations = np.array([], dtype='int')

sino_rrmse_array = np.array([])
dyn_sino_rrmse_array = np.array([])

obj_rrmse_array = np.array([])
dyn_obj_rrmse_array = np.array([])

channels_dist_mean = np.array([])
channels_dist_median = np.array([])
channels_dist_std = np.array([])

gap_array = np.array([], dtype='int')


iteration_stop_diff = 0.001

# number_of_tests = 1000
number_of_tests = 100

for ii in np.arange(number_of_tests):

    gap = np.random.choice(max_gap + 1)

    print(f'iteration {ii}, gap {gap}')
    empty_object, mask_object, channels_pixels, channels_stats = _fn.generate_random_channels_with_exact_gap(blank_object, 
                                                                                                             ch_radius, 
                                                                                                             number_of_ch, 
                                                                                                             gap=gap)
    if len(channels_pixels) != number_of_ch:
        continue
    else:
        phantom_data = _fn.create_random_phantom_objects(number_of_time_points, 
                                                         empty_object, 
                                                         mask_object, 
                                                         channels_pixels, 
                                                         size)
        result = _fn.calc_2sts_4dct(empty_object, 
                                    mask_object, 
                                    *phantom_data[:-1], 
                                    number_of_time_points, 
                                    alg_type='new', 
                                    show_info=False, 
                                    iteration_stop_diff=iteration_stop_diff)
        
        channels_dist_mean = np.append(channels_dist_mean, channels_stats['mean'])
        channels_dist_median = np.append(channels_dist_median, channels_stats['median'])
        
        n_of_iterations = np.append(n_of_iterations, result['iterations'])
        
        sino_rrmse_array = np.append(sino_rrmse_array, result['sino_rrmse'][-1])
        dyn_sino_rrmse_array = np.append(dyn_sino_rrmse_array, result['dyn_sino_rrmse'])
        
        obj_rrmse_array = np.append(obj_rrmse_array, result['obj_rrmse'][-1])
        dyn_obj_rrmse_array = np.append(dyn_obj_rrmse_array, result['dyn_obj_rrmse'])
        
        gap_array = np.append(gap_array, gap)


# %% [markdown]
# ### save collected data to file

# %%
data_to_save = np.stack((
    channels_dist_mean, 
    channels_dist_median, 
    n_of_iterations, 
    sino_rrmse_array, 
    obj_rrmse_array, 
    gap_array,
    dyn_sino_rrmse_array, 
    dyn_obj_rrmse_array, 
))

with open('bulk_data.npy', 'wb') as f:
    np.save(f, data_to_save)

# %% [markdown]
# ### load collected data from file (use file with 1000 number of tests)

# %%
# bulk_data_file = 'bulk_data_8_8_1000.npy'
# bulk_data_file = 'bulk_data_16_4_1000.npy'
bulk_data_file = 'bulk_data_32_2_1000.npy'

with open(bulk_data_file, 'rb') as f:
    (channels_dist_mean, 
    channels_dist_median, 
    n_of_iterations, 
    sino_rrmse_array, 
    obj_rrmse_array, 
    gap_array,
    dyn_sino_rrmse_array, 
    dyn_obj_rrmse_array) = np.load(f)

is_nan_idx = np.isnan(sino_rrmse_array)
print(is_nan_idx.sum())

if is_nan_idx.sum() > 0:
    channels_dist_mean = channels_dist_mean[~is_nan_idx]
    channels_dist_median = channels_dist_median[~is_nan_idx]
    n_of_iterations = n_of_iterations[~is_nan_idx]
    sino_rrmse_array = sino_rrmse_array[~is_nan_idx]
    obj_rrmse_array = obj_rrmse_array[~is_nan_idx]
    gap_array = gap_array[~is_nan_idx]
    dyn_sino_rrmse_array = dyn_sino_rrmse_array[~is_nan_idx]
    dyn_obj_rrmse_array = dyn_obj_rrmse_array[~is_nan_idx]

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
im0 = ax[0].scatter(channels_dist_mean, sino_rrmse_array, marker='.', c=n_of_iterations)
im1 = ax[1].scatter(channels_dist_mean, obj_rrmse_array, marker='.', c=n_of_iterations)
ax[0].set_xlabel('$d_{mean}$', fontsize=18)
ax[1].set_xlabel('$d_{mean}$', fontsize=18)
ax[0].set_ylabel('$RRMSE_S$', fontsize=18)
ax[1].set_ylabel('$RRMSE_V$', fontsize=18)
fig.colorbar(im0, ax=ax[0], location='top', label='iterations')
fig.colorbar(im1, ax=ax[1], location='top', label='iterations')
ax[0].grid()
ax[1].grid()

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
im0 = ax[0].scatter(channels_dist_mean, dyn_sino_rrmse_array, marker='.', c=n_of_iterations)
im1 = ax[1].scatter(channels_dist_mean, dyn_obj_rrmse_array, marker='.', c=n_of_iterations)
ax[0].set_xlabel('$d_{mean}$', fontsize=18)
ax[1].set_xlabel('$d_{mean}$', fontsize=18)
ax[0].set_ylabel('$RRMSE_{S_D}$', fontsize=18)
ax[1].set_ylabel('$RRMSE_{V_D}$', fontsize=18)
fig.colorbar(im0, ax=ax[0], location='top', label='iterations')
fig.colorbar(im1, ax=ax[1], location='top', label='iterations')
ax[0].grid()
ax[1].grid()

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
im0 = ax[0].scatter(n_of_iterations, sino_rrmse_array, marker='.', c=channels_dist_mean)
im1 = ax[1].scatter(n_of_iterations, obj_rrmse_array, marker='.', c=channels_dist_mean)
ax[0].set_xlabel('iterations', fontsize=14)
ax[1].set_xlabel('iterations', fontsize=14)
ax[0].set_ylabel('$RRMSE_S$', fontsize=18)
ax[1].set_ylabel('$RRMSE_V$', fontsize=18)
fig.colorbar(im0, ax=ax[0], location='top', label='$d_{mean}$')
fig.colorbar(im1, ax=ax[1], location='top', label='$d_{mean}$')
ax[0].grid()
ax[1].grid()

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
im0 = ax[0].scatter(n_of_iterations, dyn_sino_rrmse_array, marker='.', c=channels_dist_mean)
im1 = ax[1].scatter(n_of_iterations, dyn_obj_rrmse_array, marker='.', c=channels_dist_mean)
ax[0].set_xlabel('iterations', fontsize=14)
ax[1].set_xlabel('iterations', fontsize=14)
ax[0].set_ylabel('$RRMSE_{S_D}$', fontsize=18)
ax[1].set_ylabel('$RRMSE_{V_D}$', fontsize=18)
fig.colorbar(im0, ax=ax[0], location='top', label='$d_{mean}$')
fig.colorbar(im1, ax=ax[1], location='top', label='$d_{mean}$')
ax[0].grid()
ax[1].grid()

# %%
plt.scatter(obj_rrmse_array, sino_rrmse_array, marker='.', c=channels_dist_mean)
plt.xlabel('$RRMSE_V$', fontsize=18)
plt.ylabel('$RRMSE_S$', fontsize=18)
plt.colorbar(location='top', label='$d_{mean}$')
plt.grid()

np.corrcoef(obj_rrmse_array, sino_rrmse_array)[0, 1]

# %%
plt.scatter(dyn_obj_rrmse_array, dyn_sino_rrmse_array, marker='.', c=channels_dist_mean)
plt.xlabel('$RRMSE_{V_D}$', fontsize=18)
plt.ylabel('$RRMSE_{S_D}$', fontsize=18)
plt.colorbar(location='top', label='$d_{mean}$')
plt.grid()

np.corrcoef(dyn_obj_rrmse_array, dyn_sino_rrmse_array)[0, 1]

# %%
