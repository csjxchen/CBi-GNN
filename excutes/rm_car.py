import os
det_path = '../experiments/sassd_reproduce/multi_cfg_cbignn_pswarp/48_outs_result'
file_list_dir = '/chenjiaxin/research/PointRCNN/data/KITTI/ImageSets/test.txt'
result_path = '../experiments/sassd_reproduce/multi_cfg_cbignn_pswarp/48_outs_result_rm_cars'
os.makedirs(result_path, exist_ok=True)
with open(file_list_dir, 'r') as f:
    sample_ids = list(map(int, f.read().splitlines()))

for sample_id in sample_ids:
    lines = [line.rstrip() for line in open(os.path.join(det_path, "%06d.txt" %sample_id))]
    lines = [line for line in lines if line.split()[0]!='Car']
    save_file = os.path.join(result_path, "%06d.txt" % sample_id)
    f = open(save_file, 'w+')
    if len(lines) != 0 :
        for l in lines:
            f.write(l + '\n')
    else:
        f.close()


    # lines = []
