import os
import io
import glob
import multiprocessing as mp
from enum import Enum
from datetime import datetime

import tensorflow as tf
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from google.cloud import storage

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

from utils import get_console_logger

logger = get_console_logger()

# Config
image_downsize_factor = 4
num_images = 8
writer_batch_size = 64
src_dir = "/home/adeel/data/waymo_open_dataset_end_to_end_camera_v_1_0_0"
dst_dir = "/home/adeel/data/waymo_open_dataset_end_to_end_camera_v_1_0_0_parquet_v0_1"
split_type = "val"  # "train", "test", val
num_processes = mp.cpu_count() // 2

if split_type in ["test", "val"]:
    dst_dir += f"_{split_type}"

os.makedirs(src_dir, exist_ok=True)
os.makedirs(dst_dir, exist_ok=True)

logger.info(f"---Number of Processes--- {num_processes}")

gsutil_download_commands = []
for i in range(0, 9 + 1):
    if i < 10:
        file_num = "000" + str(i)
    else:
        file_num = "00" + str(i)
    command = f"gsutil -m cp -R gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/{split_type}\*tfrecord-{file_num}\* {src_dir}/"
    gsutil_download_commands.append(command)


# waymo-open-dataset/src/waymo_open_dataset/dataset.proto -> CameraName
class CameraName(Enum):
    FRONT = 1
    FRONT_LEFT = 2
    FRONT_RIGHT = 3
    SIDE_LEFT = 4
    SIDE_RIGHT = 5
    REAR_LEFT = 6
    REAR = 7
    REAR_RIGHT = 8


camera_fields = []
for i in range(1, num_images + 1):
    camera_name = "IMAGE_" + str(CameraName(i).value) + "_" + CameraName(i).name
    camera_fields.append(pa.field(camera_name, pa.binary()))
schema = pa.schema(
    [
        pa.field("name", pa.string()),  # data.frame.context.name
        pa.field("past_states_pos", pa.list_(pa.float32())),
        pa.field("past_states_vel", pa.list_(pa.float32())),
        pa.field("past_states_accel", pa.list_(pa.float32())),
        pa.field("future_states_pos", pa.list_(pa.float32())),
        pa.field("intent", pa.uint8()),
    ]
    + camera_fields
)
logger.info(f"---Schema---")
logger.info(schema)


def worker_function(payload):
    worker_n, src_files_chunk = payload
    worker_n += 1

    if not src_files_chunk:
        return

    for src_file_idx, src_file in enumerate(src_files_chunk):

        file_num = int(src_file.split("-")[-3])

        split_type = "train"
        if "test" in src_file:
            split_type = "test"
        elif "val" in src_file:
            split_type = "val"

        dst_file = dst_dir + "/" + split_type + "_" + str(file_num) + ".parquet"
        logger.info(
            f"{worker_n} ---Processing File--- {src_file_idx} / {len(src_files_chunk)}"
        )
        logger.info(f"{worker_n} src_file = {src_file}")
        logger.info(f"{worker_n} dst_file = {dst_file}")

        dataset = tf.data.TFRecordDataset([src_file], compression_type="")
        dataset_iter = dataset.as_numpy_iterator()

        writer = pq.ParquetWriter(dst_file, compression="GZIP", schema=schema)
        records = []

        for data_idx, bytes_example in enumerate(dataset_iter):
            data = wod_e2ed_pb2.E2EDFrame()
            data.ParseFromString(bytes_example)

            record = {}
            record["name"] = data.frame.context.name
            record["past_states_pos"] = (
                np.stack([data.past_states.pos_x, data.past_states.pos_y], axis=1)
                .flatten()
                .tolist()
            )
            record["past_states_vel"] = (
                np.stack([data.past_states.vel_x, data.past_states.vel_y], axis=1)
                .flatten()
                .tolist()
            )
            record["past_states_accel"] = (
                np.stack([data.past_states.accel_x, data.past_states.accel_y], axis=1)
                .flatten()
                .tolist()
            )
            record["future_states_pos"] = (
                np.stack([data.future_states.pos_x, data.future_states.pos_y], axis=1)
                .flatten()
                .tolist()
            )
            record["intent"] = data.intent
            for image_content in data.frame.images:
                if image_content.name == 0:
                    continue
                image = tf.io.decode_image(image_content.image).numpy()
                downsized_shape = [
                    image.shape[0] // image_downsize_factor,
                    image.shape[1] // image_downsize_factor,
                ]
                downsized_image = (
                    tf.image.resize(image, downsized_shape, method="bicubic")
                    .numpy()
                    .astype(np.uint8)
                )
                compressed_image = io.BytesIO()
                np.savez_compressed(compressed_image, downsized_image)
                camera_name = CameraName(image_content.name).name + str(
                    image_content.name
                )
                camera_name = (
                    "IMAGE_"
                    + str(image_content.name)
                    + "_"
                    + CameraName(image_content.name).name
                )
                record[camera_name] = compressed_image.getvalue()

            records.append(record)

            if data_idx != 0 and data_idx % writer_batch_size == 0:
                logger.info(f"{worker_n} Writing batch {data_idx}")
                batch = pa.RecordBatch.from_pylist(records, schema=schema)
                writer.write_batch(batch)
                records = []

        if records:
            batch = pa.RecordBatch.from_pylist(records, schema=schema)
            writer.write_batch(batch)
            records = []

        writer.close()

        logger.info(f"{worker_n} ---Done--- {src_file_idx} / {len(src_files_chunk)}")

    logger.info(f"{worker_n} ---Done---")


if __name__ == "__main__":
    logger.info(f"---Start---")

    for gsutil_command in gsutil_download_commands:
        logger.info(f"---Downloading Files from GCS---")
        logger.info(gsutil_command)
        os.system(f"rm -fr {src_dir}/*")
        os.system(gsutil_command)

        src_files = sorted(
            glob.glob(src_dir + "/*"), key=lambda x: int(x.split("-")[-3])
        )
        chunk_size = len(src_files) // num_processes
        remainder = len(src_files) % num_processes
        src_files_chunks = []
        start_index = 0
        for i in range(num_processes):
            end_index = start_index + chunk_size + (1 if i < remainder else 0)
            src_files_chunks.append((i, src_files[start_index:end_index]))
            start_index = end_index
        logger.info(f"---File Chunks--- {len(src_files_chunks)} chunks")
        for i, chunk in enumerate(src_files_chunks):
            logger.info(f"Chunk {i}: {len(chunk[1])} files")
            for file in chunk[1]:
                logger.info(f"{file}")

        pool = mp.Pool(processes=num_processes)
        pool.map(worker_function, src_files_chunks)
        pool.close()

    logger.info(f"---All Done---")
