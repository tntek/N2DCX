# source domain: visda-c
~/anaconda3/bin/python N2DC_source.py --trte val --output ckps2020r0/source/ --da uda --gpu_id 0 --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 10 --s 0
# target domain: visda-c
~/anaconda3/bin/python N2DC_target.py --cls_par 0.2 --da uda --dset VISDA-C --gpu_id 0 --s 0 --t 1 --output_src ckps2020r0/source/ --output ckps2020r0/target_n2dc/ --net resnet101 --lr 1e-3 --seed 2020
~/anaconda3/bin/python N2DCEX_target.py --cls_par 0.2 --da uda --dset VISDA-C --gpu_id 0 --s 0 --t 1 --output_src ckps2020r0/source/ --output ckps2020r0/target_n2dcex/ --net resnet101 --lr 1e-3 --seed 2020


# source domain: office-31
~/anaconda3/bin/python N2DC_source.py --trte val --da uda --output ckpsoc/source/ --gpu_id 0 --dset office --max_epoch 100 --s 0
# target domain: office-31
~/anaconda3/bin/python N2DC_target.py --cls_par 0.2 --da uda --output_src ckpsoc/source/ --output ckpsoc/target_n2dc/ --gpu_id 0 --dset office --s 0  --t 1
~/anaconda3/bin/python N2DCEX_target.py --cls_par 0.2 --da uda --output_src ckpsoc/source/ --output ckpsoc/target_n2dcex/ --gpu_id 0 --dset office --s 0  --t 1


# source domain: office-home
~/anaconda3/bin/python N2DC_source.py --trte val --da uda --output ckpsoh/source/ --gpu_id 0 --dset office-home --max_epoch 100 --s 0
# target domain: office-home
~/anaconda3/bin/python N2DC_target.py --cls_par 0.2 --da uda --output_src ckpsoh/source/ --output ckpsoh/target_n2dc/ --gpu_id 0 --dset office-home --s 0  --t 1
~/anaconda3/bin/python N2DCEX_target.py --cls_par 0.2 --da uda --output_src ckpsoh/source/ --output ckpsoh/target_n2dcex/ --gpu_id 0 --dset office-home --s 0  --t 1
