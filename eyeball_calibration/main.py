import os
import argparse
from solver_multigaze_colmap import ColmapGazeSolver
from dataloader import get_colmap_loader
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/single_image.yaml', help='path to configure file')
    parser.add_argument('--outdir', type=str, default='outputs', help='output directory')
    parser.add_argument('--name', type=str, default='default_name', help='experiment name')
    parser.add_argument('--mode', type=str, default='fit', help='[fit, intergaze, seg]')
    parser.add_argument('--bg', default=None, help='background image')
    args = parser.parse_args()

    # make dir and copy config
    outinfo = utils.smart_mkdir(args.outdir, args.name)
    os.system("cp %s %s" % (args.config, outinfo[0]))
    config = utils.load_yaml(args.config)

    if 'init_npz' in config['optim'].keys():
        if config['optim']['init_npz'] == -1:
            init_dir = os.path.join(args.outdir, args.name + '_init')   # check whether "+ '_init'" is consistent with run.sh
            config['optim']['init_npz'] = os.path.join(init_dir, 'parameters.npz')
        assert isinstance(config['optim']['init_npz'], str)

    train_dataloader, test_dataloader, dataset = get_colmap_loader(config)
    solver = ColmapGazeSolver(config, outinfo, dataset)
    if args.mode == 'intergaze':
        assert args.bg is not None
        solver.save_interploate_both(dataset, args.bg, inter_mode='ud')
        solver.save_interploate_both(dataset, args.bg, inter_mode='lr')
        solver.save_interploate_both(dataset, args.bg, inter_mode='circle')
    elif args.mode == 'seg':
        solver.save_seg(411)
    else:
        solver.solve_proj(train_dataloader, test_dataloader)
        solver.solve_pose(train_dataloader, test_dataloader)
        solver.save_results(test_dataloader)
