import argparse
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(root)
os.environ['rvc_root'] = root
def set_env():
    os.environ['model_root'] = os.path.join(os.getenv('rvc_root'), 'models')
    os.environ['weight_root'] = os.path.join(os.getenv('model_root'), 'weights')
    os.environ['config_root'] = os.path.join(os.getenv('rvc_root'), 'configs')
    os.environ['rmvpe_root'] = os.path.join(os.getenv('model_root'), 'rmvpe')


####
# USAGE
#
# In your Terminal or CMD or whatever


def arg_parse():
    global root
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=str, help="input path")
    parser.add_argument("-m", "--model", type=str, help="model name / store in models/weight_root")
    parser.add_argument("-t", "--transpose", type=int, default=0)
    parser.add_argument("--index", type=str, help="index path")
    parser.add_argument("-f", "--f0-method", type=str, default="rmvpe", help="pm, harvest, crepe, or rmvpe. default: rmvpe")
    parser.add_argument("--output", type=str, help="output path", default='')
    parser.add_argument("-r", "--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--half", action='store_true', help="use half -> True")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sr")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="rms mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect")
    parser.add_argument("-l", "--list-models", action='store_true', help='Show installed models')
    if len(sys.argv) == 1:
        sys.argv.append('--help')
    args = parser.parse_args()
    sys.argv = sys.argv[:1]


    if args.list_models:
        models = [f for f in os.listdir(os.getenv('weight_root')) if f.endswith('.pth') or f.endswith('.pt')]
        if len(models) >= 1:
            print("Installed weights:")
            for i in models:
                b = os.path.splitext(os.path.basename(i))[0]
                print(f"- {b}")
        else:
            print("No weights installed.")
        sys.exit(0)
    if args.input is None:
        print("error: audio file argument required", file=sys.stderr)
        sys.exit(1)
    args.input = os.path.abspath(args.input)
    if args.output == '' and args.input != '':
          dir = os.path.dirname(args.input)
          base = os.path.basename(args.input)
          basename, _ = os.path.splitext(base)
          args.output = os.path.join(dir, f'{basename}_by_{args.model}.wav')
    os.environ['index_root'] = os.path.join(os.getenv('weight_root'), args.model + '_index')

    return args
def main():
    set_env()
    args = arg_parse()
    from dotenv import load_dotenv
    from scipy.io import wavfile
    from configs.config import Config
    from infer.modules.vc.modules import VC
    load_dotenv()
    config = Config()
    config.device = args.device if args.device else config.device
    config.is_half = args.half if args.half else config.is_half
    vc = VC(config)
    weight_path = f"{args.model}.pth"
    vc.get_vc(weight_path)
    _, wav_output = vc.vc_single(
        0,
        args.input,
        args.transpose,
        None,
        args.f0_method,
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
