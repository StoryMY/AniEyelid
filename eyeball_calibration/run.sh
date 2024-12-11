case=0111_srt_crop
savedir=outputs

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config configs/init_os.yaml \
    --name ${case}_os_init \
    --outdir ${savedir}

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config configs/optim_os.yaml \
    --name ${case}_os \
    --outdir ${savedir}

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config configs/init_od.yaml \
    --name ${case}_od_init \
    --outdir ${savedir}

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config configs/optim_od.yaml \
    --name ${case}_od \
    --outdir ${savedir}

