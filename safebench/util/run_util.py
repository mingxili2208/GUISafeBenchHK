import os
import os.path as osp
import time
import json
import re
import numpy as np
from fnmatch import fnmatch
import yaml
import importlib
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


class VideoWriter:
    def __init__(self, filename='_autoplay.mp4', fps=10.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint64(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        # self.writer.write_frame(img)
        try:
            self.writer.write_frame(img)
        except:
            pass

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()


class VideoRecorder(object):
    def __init__(self, output_dir, logger):
        self.logger = logger
        self.output_dir = output_dir
        self.video_count = 0
        self.fps = 20
        self.frame_list = []
        hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.video_dir = os.path.join(self.output_dir, 'video', hms_time)

    def add_frame(self, frame):
        self.frame_list.append(frame)

    @staticmethod
    def _sanitize_token(value):
        token = re.sub(r'[^A-Za-z0-9._-]+', '-', str(value)).strip('-_.')
        return token or 'na'

    @staticmethod
    def _format_identifier(values, width=2):
        formatted = []
        for value in values or []:
            if isinstance(value, int):
                formatted.append(f'{value:0{width}d}')
                continue

            value_str = str(value)
            if value_str.isdigit():
                formatted.append(f'{int(value_str):0{width}d}')
            else:
                formatted.append(VideoRecorder._sanitize_token(value_str))

        return '-'.join(formatted) if formatted else 'na'

    def _build_video_name(self, batch_metadata):
        data_part = self._format_identifier(batch_metadata.get('data_ids', []), width=4)
        scenario_part = self._format_identifier(batch_metadata.get('scenario_ids', []), width=2)
        route_part = self._format_identifier(batch_metadata.get('route_ids', []), width=2)
        content_tag = self._sanitize_token(batch_metadata.get('content_tag', 'eval'))

        return (
            f'video_{self.video_count:04d}'
            f'_data_{data_part}'
            f'_scen_{scenario_part}'
            f'_route_{route_part}'
            f'_{content_tag}.mp4'
        )

    def save(self, data_ids=None, scenario_names=None, batch_metadata=None):
        if batch_metadata is None:
            batch_metadata = {
                'data_ids': data_ids or [],
                'scenario_names': scenario_names or [],
                'scenario_ids': [],
                'route_ids': [],
                'content_tag': 'planning',
            }

        video_name = self._build_video_name(batch_metadata)
        os.makedirs(self.video_dir, exist_ok=True)
        video_file = os.path.join(self.video_dir, video_name)
        self.logger.log(f'>> Saving video to {video_file}')

        # define video writer
        video_writer = VideoWriter(filename=video_file, fps=self.fps)
        for f in self.frame_list:
            video_writer.add(f)
        video_writer.close()

        metadata = dict(batch_metadata)
        metadata.update({
            'video_index': self.video_count,
            'video_file': video_file,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'frame_count': len(self.frame_list),
        })
        with open(video_file + '.json', 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

        # reset frame list
        self.frame_list = []
        self.video_count += 1
        return video_file


class VideoRecorder_Perception(object):
    def __init__(self, output_dir, logger, width=1024, height=1024):
        self.logger = logger
        self.output_dir = output_dir
        self.video_dir = os.path.join(self.output_dir, 'video')
        self.video_count = 0
        
        self.frame_list = []
        # TODO: parse observation size
        self.width, self.height = width, height

    def add_frame(self, frame):
        self.frame_list.append(frame)

    def save(self, data_ids=None, scenario_names=None, batch_metadata=None):
        if batch_metadata is None:
            batch_metadata = {
                'data_ids': data_ids or [],
                'scenario_names': scenario_names or [],
                'scenario_ids': [],
                'route_ids': [],
                'content_tag': 'perception',
            }

        data_ids = batch_metadata.get('data_ids', [])
        scenario_ids = batch_metadata.get('scenario_ids', [])
        route_ids = batch_metadata.get('route_ids', [])
        num_episodes = len(data_ids)
        os.makedirs(self.video_dir, exist_ok=True)

        video_name = []
        for idx, data in enumerate(data_ids):
            scenario_id = scenario_ids[idx] if idx < len(scenario_ids) else 'na'
            route_id = route_ids[idx] if idx < len(route_ids) else 'na'
            video_name.append(
                f'video_{self.video_count:04d}'
                f'_data_{int(data):04d}'
                f'_scen_{VideoRecorder._format_identifier([scenario_id], width=2)}'
                f'_route_{VideoRecorder._format_identifier([route_id], width=2)}'
                f'_perception.mp4'
            )
        video_file = [os.path.join(self.video_dir, v) for v in video_name]
        self.logger.log(f'>> Saving video to {self.video_dir}')
        self.writer_list = [VideoWriter(filename=v, fps=20.0) for v in video_file]
        for f in self.frame_list:
            for n_i in range(num_episodes): 
                try:
                    self.writer_list[n_i].add(f[n_i])
                except:
                    pass
        for n_i in range(num_episodes):
            self.writer_list[n_i].close()

        for n_i, current_video_file in enumerate(video_file):
            metadata = dict(batch_metadata)
            metadata.update({
                'video_index': self.video_count,
                'video_file': current_video_file,
                'data_id': data_ids[n_i],
                'scenario_id': scenario_ids[n_i] if n_i < len(scenario_ids) else None,
                'route_id': route_ids[n_i] if n_i < len(route_ids) else None,
                'scenario_name': (scenario_names[n_i] if scenario_names and n_i < len(scenario_names) else None),
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            })
            with open(current_video_file + '.json', 'w', encoding='utf-8') as metadata_file:
                json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        
        self.logger.log(f'>> Saving video done.')
        self.frame_list = []
        self.video_count += 1
        return video_file


def print_dict(d):
    print(yaml.dump(d, sort_keys=False, default_flow_style=False))


# ->dict 是函数注解的一部分,用于指示函数的返回类型
def load_config(config_path="default_config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_config_dir(dir, depth=0):
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name == "config.yaml":
                return path, name
    # if we can not find the config file from the current dir, we search for the parent dir:
    if depth > 2:
        return None
    return find_config_dir(osp.dirname(dir), depth + 1)


def find_model_path(dir, itr=None):
    # if itr is specified, return model with the itr number
    if itr is not None:
        model_path = osp.join(dir, "model_" + str(itr) + ".pt")
        if not osp.exists(model_path):
            return None
            # raise ValueError("Model doesn't exist: " + model_path)
        return model_path
    # if itr is not specified, return model.pt or the one with the largest itr number
    pattern = "*pt"
    model = "model.pt"
    max_itr = -1
    for _, _, files in os.walk(dir):
        for name in files:
            if fnmatch(name, pattern):
                name = name.split(".pt")[0].split("_")
                if len(name) > 1:
                    itr = int(name[1])
                    if itr > max_itr:
                        max_itr = itr
                        model = "model_" + str(itr) + ".pt"
    model_path = osp.join(dir, model)
    if not osp.exists(model_path):
        return None
        # raise ValueError("Model doesn't exist: " + model_path)
    return model_path, max_itr


def setup_eval_configs(dir, itr=None):
    path, config_name = find_config_dir(dir)
    model_path, load_itr = find_model_path(osp.join(path, "model_save"), itr=itr)
    config_path = osp.join(path, config_name)
    configs = load_config(config_path)
    return model_path, load_itr, configs["policy"], configs["timeout_steps"], configs[configs["policy"]]


def class_from_path(path):
    # 从路径中提取模块名和类名，然后返回类对象
    module_name, class_name = path.rsplit(".", 1)  # 从右边开始分割，分割一次
    class_object = getattr(importlib.import_module(module_name), class_name)  # 通过模块名和类名获取类对象
    return class_object
