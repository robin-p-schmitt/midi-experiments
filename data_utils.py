import subprocess
import os
import h5py
import numpy as np
import pretty_midi
from mido import KeySignatureError

DATA_BASE_PATH = 'data'
DATA_GZ_NAME = 'lmd_full.tar.gz'
DATA_GZ_PATH = os.path.join(DATA_BASE_PATH, DATA_GZ_NAME)
DATA_UNPACKED_PATH = os.path.join(DATA_BASE_PATH, 'lmd_full')
HDF_FILE_PATH = os.path.join(DATA_BASE_PATH, 'piano_rolls.hdf5')

MAX_NUM_TIME_STEPS = 500
TARGET_INSTRUMENT = "Bass"
MAX_NUM_PROCESSED_FILES = 5_000
START_NOTE = 21
NUM_NOTES = 88


def download_dataset():
  if os.path.exists(DATA_GZ_PATH):
    print('Dataset already downloaded.')
    return

  print('Downloading dataset...')
  os.makedirs(DATA_BASE_PATH, exist_ok=True)
  subprocess.run(['wget', 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz', '-O', DATA_GZ_PATH])
  print('Downloaded dataset.')


def unpack_dataset():
  if os.path.exists(DATA_UNPACKED_PATH):
    print('Dataset already unpacked.')
    return

  print('Unpacking dataset...')
  subprocess.run(['tar', 'xvzf', DATA_GZ_PATH, '-C', DATA_BASE_PATH])
  print('Unpacked dataset.')


def extract_piano_roll_from_instrument(instrument: pretty_midi.Instrument):
  """
  Extract the piano roll for the target instrument and binarize it.
  :param instrument:
  :return:
  """
  # Extract the piano roll for the target instrument
  piano_roll = instrument.get_piano_roll(fs=100)[START_NOTE:START_NOTE + NUM_NOTES]  # (NUM_NOTES, MAX_NUM_TIME_STEPS)
  if np.all(piano_roll == 0) or piano_roll.shape[1] < MAX_NUM_TIME_STEPS:
    # Skip empty piano rolls and those shorter than MAX_NUM_TIME_STEPS
    return None

  # roll array such that the instrument starts in the first time step
  first_non_zero_col = np.nonzero(piano_roll)[1][0]
  piano_roll = np.roll(piano_roll, -first_non_zero_col)
  # clip the time to time_steps
  piano_roll = piano_roll[:, :MAX_NUM_TIME_STEPS]
  piano_roll[piano_roll > 0] = 1  # binarize the piano roll
  return piano_roll


def build_hdf_file():
  if os.path.exists(HDF_FILE_PATH):
    print('HDF file already exists.')
    return

  # create empty, expandable hdf5 file
  with h5py.File(HDF_FILE_PATH, "w") as hf:
    # Create an expandable dataset for piano rolls
    hf.create_dataset(
      "piano_rolls",
      shape=(0, NUM_NOTES, MAX_NUM_TIME_STEPS),
      maxshape=(None, NUM_NOTES, MAX_NUM_TIME_STEPS),  # Unlimited growth along the first dimension
      dtype=np.float32
    )

    # Process MIDI files and append to the dataset
    num_processed_files = 0
    for dirpath, dirnames, filenames in os.walk(DATA_UNPACKED_PATH):
      if num_processed_files >= MAX_NUM_PROCESSED_FILES:
        break
      for midi_filename in filenames:
        num_processed_files += 1
        midi_path = os.path.join(dirpath, midi_filename)

        if num_processed_files % 1000 == 0:
          print(f"Processed {num_processed_files} midi files")
        if num_processed_files >= MAX_NUM_PROCESSED_FILES:
          break

        # Load the MIDI file
        try:
          pm = pretty_midi.PrettyMIDI(midi_path)
        # some files are known to be corrupted, so we skip them
        except (OSError, EOFError, KeySignatureError, ValueError, KeyError, IndexError) as e:
          print(f"Error loading {midi_path}: {e}")
          continue

        # only process midi files which include target instrument
        instruments = [instrument for instrument in pm.instruments if TARGET_INSTRUMENT in instrument.name]
        if len(instruments) > 0:
          instrument = instruments[0]
        else:
          continue

        # Extract the piano roll for the target instrument
        piano_roll = extract_piano_roll_from_instrument(instrument)
        if piano_roll is None:
          continue

        # Normalize piano roll values to [0, 1] (optional)
        piano_roll = np.clip(piano_roll / 127, 0, 1)

        # Resize and append the piano roll to the dataset
        # Resize the dataset to accommodate the new data
        hf["piano_rolls"].resize((hf["piano_rolls"].shape[0] + 1, NUM_NOTES, MAX_NUM_TIME_STEPS))
        hf["piano_rolls"][-1] = piano_roll  # Add new piano roll

    # Verify the final dataset
    print("Final dataset shape:", hf["piano_rolls"].shape)
