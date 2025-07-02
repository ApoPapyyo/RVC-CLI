import argparse
import os
import sys

root = os.path.dirname(os.path.dirname(sys.argv[0]))
sys.path.append(root)
from dotenv import load_dotenv
from scipy.io import wavfile

from configs.config import Config
from infer.modules.vc.modules import VC

####
# USAGE
#
# In your Terminal or CMD or whatever


def arg_parse():
    global root
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=str)
    parser.add_argument("input", type=str, help="input path")
    parser.add_argument("-m", "--model", type=str, help="model name / store in models/weight_root")
    parser.add_argument("-t", "--transpose", type=int, default=0)
    parser.add_argument("--index", type=str, help="index path")
    parser.add_argument("-f", "--f0method", type=str, default="rmvpe", help="harvest or pm")
    parser.add_argument("--output", type=str, help="output path", default='')
    parser.add_argument("-r", "--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--half", action='store_true', help="use half -> True")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sr")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="rms mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect")
    parser.add_argument("--list-models", action='store_true', help='Show installed models')

    args = parser.parse_args()
    os.chdir(args.workdir)
    args.input = os.path.abspath(args.input)
    os.chdir(root)
    sys.argv = sys.argv[:1]
    if args.output == '' and args.input != '':
          dir = os.path.dirname(args.input)
          base = os.path.basename(args.input)
          basename, _ = os.path.splitext(base)
          args.output = dir + '/' + basename + '_by_' + args.model + '.wav'

    return args


def main():
    load_dotenv()
    args = arg_parse()
    config = Config()
    config.device = args.device if args.device else config.device
    config.is_half = args.half if args.half else config.is_half
    vc = VC(config)
    sid = str(args.model)+'.pth'
    index = str(args.model)+'_index'
    os.environ['index_root'] = f'{os.getenv("weight_root")}/{index}'
    vc.get_vc(sid)
    os.environ['rmvpe_root'] = f'{os.path.dirname(os.getenv("weight_root"))}/rmvpe'
    _, wav_output = vc.vc_single(
        0,
        args.input,
        args.transpose,
        None,
        args.f0method,
        args.index,
        None,
        args.index_rate,
        args.filter_radius,
        args.resample_sr,
        args.rms_mix_rate,
        args.protect,
    )
    wavfile.write(args.output, wav_output[0], wav_output[1])


if __name__ == "__main__":
    main()
