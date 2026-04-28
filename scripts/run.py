"""
1、设置全局测试参数
2、根据命令行参数指定的路径加载agent和scenario的配置文件
3、根据runner的类型执行相应的run方法
"""

import traceback
import os.path as osp
import time
import torch
from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner import CarlaRunner


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if value in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Cannot parse boolean value: {value}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')

    # 定义测试结果的输出目录
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    parser.add_argument('--max_episode_step', type=int, default=2000)
    parser.add_argument('--auto_ego', type=str2bool, default=False)

    # 提供三种模式选择：训练agent、训练scenario、evaluation
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_agent', 'train_scenario', 'eval'])
    parser.add_argument('--agent_cfg', nargs='+', type=str, default=['behavior.yaml'])
    parser.add_argument('--scenario_cfg', nargs='+', type=str, default=['standard.yaml'])
    parser.add_argument('--continue_agent_training', '-cat', type=str2bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=str2bool, default=False)

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', type=str2bool, default=True)
    parser.add_argument('--render', type=str2bool, default=True)

    # 每间隔多少帧再计算一次，数值太大会导致计算不及时，一直使用之前计算得到的数据进行控制
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2000, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8000, help='traffic manager port')
    # carla world中每一帧的时间间隔
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1, help='time for each frame')

    args = parser.parse_args()
    args_dict = vars(args)

    err_list = []
    for agent_cfg in args.agent_cfg:
        for scenario_cfg in args.scenario_cfg:
            # set global parameters
            set_torch_variable(args.device)
            torch.set_num_threads(args.threads)
            set_seed(args.seed)

            # load agent config
            agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', agent_cfg)
            agent_config = load_config(agent_config_path)

            # load scenario config
            scenario_config_path = osp.join(args.ROOT_DIR, 'safebench/scenario/config', scenario_cfg)
            scenario_config = load_config(scenario_config_path)

            # main entry with a selected mode
            agent_config.update(args_dict)
            scenario_config.update(args_dict)
            runner = CarlaRunner(agent_config, scenario_config)

            # start running
            start_time = time.time()
            try:
                runner.run()
            except Exception:
                traceback.print_exc()
                try:
                    runner.persist_eval_progress(reason='exception')
                except Exception:
                    traceback.print_exc()
                # agent_cfg: 当前被测试的agent配置文件; scenario_cfg: 生成测试场景的配置文件; traceback: 错误信息
                err_list.append([agent_cfg, scenario_cfg, traceback.format_exc()])
            finally:
                try:
                    runner.close()
                except Exception:
                    traceback.print_exc()
            end_time = time.time()
            print(f"Total time for {agent_cfg} and {scenario_cfg}: {end_time - start_time} seconds")



    # 运行完毕关闭CARLA
    # os.system('pkill -f "CarlaUE4"')

    # 删除log文件下的所有内容
    # os.system('rm -r ../log/*')

    for err in err_list:
        print(err[0], err[1], 'failed!')
        print(err[2])
