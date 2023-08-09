import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_bkg_data(input_file, output_file, chunk_size=100000):
    # Create an instance of the standard scaler outside the loop
    pt_scaler = StandardScaler()

    with h5py.File(input_file, 'r') as h5f_input, h5py.File(output_file, 'w') as h5f_output:
        total_samples = h5f_input['Particles'].shape[0]

        # Create datasets in the output file
        data_dset = h5f_output.create_dataset('data', (total_samples, 57), dtype='float32')
        data_target_dset = h5f_output.create_dataset('data_target', (total_samples, 57), dtype='float32')

        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)

            # Read a chunk of data
            data_chunk = h5f_input['Particles'][start_idx:end_idx, :, :-1]

            # Shuffle isn't perfect here since it's per-chunk, but it helps
            np.random.shuffle(data_chunk)

            # Scaling
            data_target_chunk = np.copy(data_chunk)
            data_target_chunk[:,:,0] = pt_scaler.partial_fit(data_target_chunk[:,:,0]).transform(data_target_chunk[:,:,0])
            data_target_chunk[:,:,0] = np.multiply(data_target_chunk[:,:,0], np.not_equal(data_chunk[:,:,0], 0))

            # Reshape
            data_chunk = data_chunk.reshape((data_chunk.shape[0], 57))
            data_target_chunk = data_target_chunk.reshape((data_target_chunk.shape[0], 57))

            # Store in the output file
            data_dset[start_idx:end_idx] = data_chunk
            data_target_dset[start_idx:end_idx] = data_target_chunk

    return pt_scaler  # Return the fitted scaler

def prepare_bsm_data(input_bsm, output_bsm, pt_scaler):
    # Read BSM data
    bsm_data = []
    for bsm_file in input_bsm:
        with h5py.File(bsm_file, 'r') as h5f_bsm:
            bsm = np.array(h5f_bsm['Particles'][:,:,:-1])
            bsm_data.append(bsm.reshape(bsm.shape[0], bsm.shape[1]*bsm.shape[2]))

    #pt_scaler = StandardScaler()
    bsm_scaled_data = []
    for bsm in bsm_data:
        bsm = bsm.reshape(bsm.shape[0], 19, 3, 1)
        bsm = np.squeeze(bsm, axis=-1)
        bsm_data_target = np.copy(bsm)
        bsm_data_target[:,:,0] = pt_scaler.transform(bsm_data_target[:,:,0])
        bsm_data_target[:,:,0] = np.multiply(bsm_data_target[:,:,0], np.not_equal(bsm[:,:,0],0))
        bsm_scaled_data.append(bsm_data_target.reshape(bsm_data_target.shape[0], bsm_data_target.shape[1]*bsm_data_target.shape[2]))

    for idx, (bsm, scaled_bsm) in enumerate(zip(bsm_data, bsm_scaled_data)):
        with h5py.File(output_bsm[idx], 'w') as h5f:
            h5f.create_dataset('data', data=bsm)
            h5f.create_dataset('scaled_data', data=scaled_bsm)
