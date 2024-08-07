import gc
import glob
import os
from datetime import timedelta
from warnings import simplefilter

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from iti.data.dataset import BaseDataset, StackDataset, get_intersecting_files
from iti.data.editor import stereo_norms, LoadMapEditor, SECCHIPrepEditor, NormalizeRadiusEditor, MapToDataEditor, \
    NormalizeEditor, BrightestPixelPatchEditor, sdo_norms, ExpandDimsEditor, Editor
from iti.translate import InstrumentToInstrument
from sunpy.map import Map
from tqdm import tqdm

# set data paths for reading and writing
prediction_path = '/mnt/nerf-data/stereo_2022_02_full_prep_converted_fov'
data_path = '/mnt/nerf-data/stereo_2022_02_full_prep'
os.makedirs(prediction_path, exist_ok=True)

# find all existing files (converted)
# existing = [os.path.basename(f) for f in glob.glob(os.path.join(prediction_path, '171', '*.fits'))]

# find all alinged stereo files (grouped by filename)
basenames_stereo = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (data_path, wl))] for
                    wl in ['171', '195', '284', '304']]
dates_stereo = [[parse(b[:-7]) for b in bns] for bns in basenames_stereo]

ref_dates = dates_stereo[-1]
ref_dates = [d for d in ref_dates if
             all([np.abs(np.array(dates) - d).min() < timedelta(minutes=5) for dates in dates_stereo])]

aligned_files_stereo = []
for dates, bns, wl in zip(dates_stereo, basenames_stereo, ['171', '195', '284', '304']):
    dates = np.array(dates)
    bns = np.array(bns)
    aligned = [bns[np.argmin(np.abs(dates - d))] for d in ref_dates]
    aligned_files = [f'{data_path}/{wl}/{bn}' for bn in aligned]
    aligned_files_stereo += [aligned_files]

# basenames_stereo = [b for b in basenames_stereo if b not in existing]

# basenames_stereo = basenames_stereo[1:]
# print device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using: %s' % str(device), torch.cuda.get_device_name(0))


class SubmapEditor(Editor):

    def __init__(self, bl_Tx, bl_Ty, tr_Tx, tr_Ty):
        self.bl_Tx = bl_Tx
        self.bl_Ty = bl_Ty
        self.tr_Tx = tr_Tx
        self.tr_Ty = tr_Ty

    def call(self, s_map, **kwargs):
        return s_map.submap(
            bottom_left=SkyCoord(self.bl_Tx, self.bl_Ty, frame=s_map.coordinate_frame),
            top_right=SkyCoord(self.tr_Tx, self.tr_Ty, frame=s_map.coordinate_frame))


# create ITI translator
class SECCHIDataset(BaseDataset):

    def __init__(self, data, wavelength, resolution=1024, degradation=None, **kwargs):
        norm = stereo_norms[wavelength]

        editors = [LoadMapEditor(),
                   SECCHIPrepEditor(degradation),
                   NormalizeRadiusEditor(resolution, crop=False),
                   SubmapEditor(-1300 * u.arcsec, -1300 * u.arcsec, 1300 * u.arcsec, 1300 * u.arcsec),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, **kwargs)


class STEREODataset(StackDataset):

    def __init__(self, data, patch_shape=None, resolution=1024, **kwargs):
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, ['171', '195', '284', '304'], **kwargs)
        data_sets = [SECCHIDataset(paths[0], 171, resolution=resolution),
                     SECCHIDataset(paths[1], 195, resolution=resolution),
                     SECCHIDataset(paths[2], 284, resolution=resolution),
                     SECCHIDataset(paths[3], 304, resolution=resolution, degradation=[-9.42497209e-05, 2.27153104e+00]),
                     ]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


class STEREOToSDO(InstrumentToInstrument):

    def __init__(self, model_name='stereo_to_sdo_v0_2.pt', **kwargs):
        super().__init__(model_name, **kwargs)

    def translate(self, path, basenames=None, return_arrays=False):
        soho_dataset = STEREODataset(path, basenames=basenames)
        for result, inputs, outputs in self._translateDataset(soho_dataset):
            norms = [sdo_norms[171], sdo_norms[193], sdo_norms[211], sdo_norms[304]]
            result = [Map(norm.inverse((s_map.data + 1) / 2), self.toSDOMeta(s_map.meta, instrument, wl))
                      for s_map, norm, instrument, wl in
                      zip(result, norms, ['AIA'] * 4, [171, 193, 211, 304])]
            if return_arrays:
                yield result, inputs, outputs
            else:
                yield result

    def toSDOMeta(self, meta, instrument, wl):
        new_meta = meta.copy()
        new_meta['obsrvtry'] = 'SOHO-to-SDO'
        new_meta['telescop'] = 'sdo'
        new_meta['instrume'] = instrument
        new_meta['WAVELNTH'] = wl
        new_meta['waveunit'] = 'angstrom'
        return new_meta


translator = STEREOToSDO(n_workers=16, device=device)

# create directories for converted data
dirs = ['171', '195', '284', '304', ]
[os.makedirs(os.path.join(prediction_path, d), exist_ok=True) for d in dirs]

# start the translation as python generator
iti_maps = translator.translate(aligned_files_stereo, basenames=basenames_stereo)

# iteratively process results
simplefilter('ignore')  # ignore int conversion warning
for iti_cube, ref_date in tqdm(zip(iti_maps, ref_dates), total=len(ref_dates)):
    for s_map, d in zip(iti_cube, dirs):
        path = os.path.join(os.path.join(prediction_path, d, ref_date.isoformat('T', timespec='minutes') + '.fits'))
        if os.path.exists(path):  # skip existing data
            continue
        s_map.save(path)  # save map to disk
        gc.collect()
